//! Muse-Glimmer dense text decoder and checkpoint adapters.

/// Muse-Glimmer DFlash external assistant.
pub mod assistant;
/// Bounded and unified residency execution for Muse-Glimmer.
pub mod layerwise;
pub(crate) mod mtp;
#[cfg(feature = "image-processing")]
pub(crate) mod processor;
pub(crate) mod vision;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::{Path, PathBuf},
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait, ModuleParametersExt},
    nn,
    ops::{
        concatenate_axis, indexing::TryIndexOp, mean_axis, rsqrt, sigmoid, tanh, GgufCheckpoint,
        GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use crate::nn::generation::sample;

use crate::{
    api::{
        common::{
            self,
            attention::{
                apply_rope_and_update_cache, apply_rotary_embeddings_and_update_cache,
                attention_probabilities, batch_seq, finish_attention, reshape_attention_projection,
                AttentionInput,
            },
            generation::CausalLm,
            layers::SwiGluMlp,
            linear::project_logits_maybe_quantized,
            moe::TopKRouterScoreFunction,
        },
        input,
    },
    error::Error,
    nn::tensor::{
        create_causal_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
    },
    runtime::attention::{AttentionPolicy, LayerSchedule},
    runtime::cache::{
        residency::{
            open_prompt_cache_snapshot, save_prompt_cache_snapshot, CacheBlockArrays,
            CacheRankIdentity, CacheResidencyManager, PromptCacheDescriptor, PromptCacheManifest,
            PromptCacheModelIdentity, PromptCacheOptions, PromptCacheSnapshotBlock,
        },
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    runtime::checkpoint::load::{
        gguf_metadata, gguf_quantization_configs, load_gguf_strict, load_named_array_strict,
        load_safetensors_dir_quantized_strict, load_safetensors_dir_strict, GgufTensorNames,
        StrictLoadConfig, StrictLoadReport,
    },
    runtime::checkpoint::quantization::WeightQuantization,
    runtime::execution::inspection::{ActivationObserver, MoeRoutingObservation},
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Source-specific weight and rotary layout.
pub enum WeightConvention {
    /// Hugging Face safetensors use centered norms and rotate-half RoPE.
    HuggingFace,
    /// Official GGUF weights are norm-shifted and Q/K-permuted.
    Gguf,
}

/// Builds the authoritative per-layer paged KV layout for any Muse-Glimmer
/// execution route.
pub(crate) fn new_paged_cache_with_manager(
    args: &DecoderConfig,
    manager: CacheResidencyManager,
    rank: Option<CacheRankIdentity>,
) -> Result<Vec<Option<PagedKeyValueCache>>, Exception> {
    args.attention_schedule
        .iter()
        .enumerate()
        .map(|(layer, policy)| {
            let window = policy.window().map(|window| {
                i32::try_from(window.get())
                    .expect("validated Muse-Glimmer attention window fits i32")
            });
            PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank).map(Some)
        })
        .collect()
}

#[derive(Debug, Clone)]
/// Validated Muse-Glimmer decoder geometry normalized from checkpoint metadata.
pub struct DecoderConfig {
    /// Model type from the configuration.
    pub model_type: String,
    /// Transformer hidden size.
    pub hidden_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,
    /// Intermediate size for the SwiGLU MLP.
    pub intermediate_size: i32,
    /// Number of query attention heads.
    pub num_attention_heads: i32,
    /// RMSNorm epsilon.
    pub rms_norm_eps: f32,
    /// Epsilon used by post-attention and post-feed-forward norms.
    pub post_norm_eps: f32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Per-layer RoPE enablement; false identifies a NoPE layer.
    pub layer_uses_rope: Vec<bool>,
    /// Per-head attention dimension. Zero is normalized from hidden/head geometry.
    pub head_dim: i32,
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    /// Optional RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Attention activation. Dense Qwen text checkpoints use SiLU.
    pub hidden_act: String,
    /// Inference checkpoints must not request attention dropout.
    pub attention_dropout: f32,
    /// Optional config declaration for attention projection bias.
    pub attention_bias: Option<bool>,
    /// Optional declaration for MLP projection biases; dense Qwen requires none.
    pub mlp_bias: Option<bool>,
    /// Authoritative attention behavior in decoder-layer order.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Multiplicative factor applied after weightless query normalization.
    pub qk_scale_factor: f32,
    /// Multiplicative scale applied to raw language-model logits.
    pub output_multiplier: f32,
    /// Positive tanh soft cap applied to scaled logits.
    pub final_logit_softcapping: f32,
    /// Source-specific normalization and rotary convention.
    pub weight_convention: WeightConvention,
    /// Preferred MLX-LM affine quantization metadata.
    pub quantization: Option<WeightQuantization>,
    /// Hugging Face-compatible alias emitted by MLX-LM converters.
    pub quantization_config: Option<WeightQuantization>,
    /// Optional exact weight names that use affine quantization.
    ///
    /// `None` preserves MLX-LM's model-wide quantization behavior. GGUF
    /// loading uses `Some` for checkpoints mixing packed and dense matrices.
    pub quantized_weights: Option<HashSet<String>>,
    /// Routed-expert intermediate size for Qwen3 MoE checkpoints.
    pub moe_intermediate_size: i32,
    /// Number of routed experts. Zero for dense Qwen3 checkpoints.
    pub num_experts: i32,
    /// Number of experts selected per token.
    pub num_experts_per_tok: i32,
    /// Whether selected routing probabilities are normalized.
    pub norm_topk_prob: bool,
    /// Per-weight affine settings for mixed GGUF Q2/Q3/Q4/Q5/Q6/Q8 tensors.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
    /// Optional native multimodal tower configuration. Official GGUF text files omit it.
    pub vision_config: Option<vision::VisionConfig>,
    /// Image placeholder token used by checkpoint-native prompts.
    pub image_token_id: u32,
    /// Video placeholder token used by checkpoint-native prompts.
    pub video_token_id: u32,
    /// Flattened pixel-shuffle width entering the vision adapter.
    pub vision_out_hidden_size: i32,
    /// Hidden width of the two-layer vision adapter.
    pub projector_hidden_size: i32,
}

fn default_hidden_act() -> String {
    "silu".into()
}

#[derive(Deserialize)]
struct DecoderConfigSource {
    model_type: String,
    hidden_size: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    rms_norm_eps: f32,
    post_norm_eps: f32,
    vocab_size: i32,
    num_key_value_heads: i32,
    max_position_embeddings: i32,
    #[serde(default)]
    rope_theta: Option<f32>,
    #[serde(default)]
    rope_parameters: HashMap<String, FloatOrString>,
    layer_types: Vec<String>,
    layer_rope_theta: Vec<f32>,
    sliding_window: i32,
    #[serde(default)]
    head_dim: i32,
    tie_word_embeddings: bool,
    rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default = "default_hidden_act", alias = "hidden_activation")]
    hidden_act: String,
    #[serde(default)]
    attention_dropout: f32,
    #[serde(default)]
    attention_bias: Option<bool>,
    #[serde(default)]
    mlp_bias: Option<bool>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantization_config: Option<WeightQuantization>,
    #[serde(default)]
    moe_intermediate_size: i32,
    #[serde(default)]
    num_experts: i32,
    #[serde(default)]
    num_experts_per_tok: i32,
    #[serde(default)]
    norm_topk_prob: bool,
    qk_scale_factor: f32,
    output_multiplier: f32,
    final_logit_softcapping: f32,
}

impl DecoderConfigSource {
    fn normalize_head_dim(&mut self) {
        if self.head_dim == 0
            && self.num_attention_heads > 0
            && self.hidden_size % self.num_attention_heads == 0
        {
            self.head_dim = self.hidden_size / self.num_attention_heads;
        }
    }

    fn into_config(
        self,
        attention_schedule: LayerSchedule<AttentionPolicy>,
        weight_convention: WeightConvention,
    ) -> DecoderConfig {
        let rope_theta = self
            .rope_theta
            .or_else(|| {
                self.rope_parameters
                    .get("rope_theta")
                    .and_then(|value| match value {
                        FloatOrString::Float(value) => Some(*value),
                        _ => None,
                    })
            })
            .unwrap_or(500_000.0);
        DecoderConfig {
            model_type: self.model_type,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            post_norm_eps: self.post_norm_eps,
            vocab_size: self.vocab_size,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta,
            layer_uses_rope: self
                .layer_rope_theta
                .iter()
                .map(|theta| *theta != 0.0)
                .collect(),
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_scaling: self.rope_scaling,
            hidden_act: self.hidden_act,
            attention_dropout: self.attention_dropout,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            attention_schedule,
            qk_scale_factor: self.qk_scale_factor,
            output_multiplier: self.output_multiplier,
            final_logit_softcapping: self.final_logit_softcapping,
            weight_convention,
            quantization: self.quantization,
            quantization_config: self.quantization_config,
            quantized_weights: None,
            moe_intermediate_size: self.moe_intermediate_size,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            norm_topk_prob: self.norm_topk_prob,
            quantized_weight_configs: None,
            vision_config: None,
            image_token_id: 200092,
            video_token_id: 200091,
            vision_out_hidden_size: 6144,
            projector_hidden_size: 4096,
        }
    }
}

impl DecoderConfig {
    pub(crate) fn model_kind(&self) -> crate::api::ModelKind {
        crate::api::ModelKind::MuseGlimmer
    }

    /// Whether this architecture carries learned Q/K/V projection biases.
    pub fn qkv_bias(&self) -> bool {
        false
    }

    /// Whether this architecture applies per-head Q/K RMS normalization.
    pub fn qk_norm(&self) -> bool {
        false
    }

    pub(crate) fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization.or(self.quantization_config)
    }

    pub(crate) fn weight_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
        if let Some(config) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|configs| configs.get(weight_name))
        {
            return Some(*config);
        }
        let quantization = self.weight_quantization()?;
        match &self.quantized_weights {
            Some(names) if !names.contains(weight_name) => None,
            _ => Some(quantization),
        }
    }

    pub(crate) fn is_moe(&self) -> bool {
        false
    }
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &DecoderConfig) -> String {
    let mut rope = args
        .rope_scaling
        .as_ref()
        .map(|config| {
            config
                .iter()
                .map(|(key, value)| {
                    let value = match value {
                        FloatOrString::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        FloatOrString::String(value) => format!("string:{value}"),
                        FloatOrString::Bool(value) => format!("bool:{value}"),
                    };
                    format!("{key}={value}")
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    rope.sort_unstable();
    format!(
        "muse-glimmer-v1:type={}:hidden={}:layers={}:q_heads={}:kv_heads={}:head_dim={}:context={}:rope_theta={:08x}:rope={}:attention={}:rope_layers={:?}:qk={:08x}",
        args.model_type,
        args.hidden_size,
        args.num_hidden_layers,
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
        args.max_position_embeddings,
        args.rope_theta.to_bits(),
        rope.join(";"),
        args.attention_schedule.fingerprint_component(),
        args.layer_uses_rope,
        args.qk_scale_factor.to_bits(),
    )
}

#[cfg(test)]
fn prompt_cache_layer_layout(
    args: &DecoderConfig,
) -> Result<LayerSchedule<crate::LayerCachePolicy>, Exception> {
    PromptCacheModelIdentity::key_value_layouts(
        args.attention_schedule.iter().map(|policy| {
            policy.window().map(|window| {
                i32::try_from(window.get()).expect("validated Muse-Glimmer window fits i32")
            })
        }),
        args.num_key_value_heads,
        args.head_dim,
    )
    .map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
fn prompt_cache_model_identity(
    args: &DecoderConfig,
) -> Result<PromptCacheModelIdentity, Exception> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Exception::custom("invalid Muse-Glimmer cache layer count"))?;
    Ok(PromptCacheModelIdentity {
        model_family: "muse_glimmer".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        sink_tokens: 0,
        topology: Default::default(),
        layer_layout: PromptCacheModelIdentity::key_value_layouts(
            args.attention_schedule.iter().map(|policy| {
                policy.window().map(|window| {
                    i32::try_from(window.get()).expect("validated Muse-Glimmer window fits i32")
                })
            }),
            args.num_key_value_heads,
            args.head_dim,
        )
        .map_err(|error| Exception::custom(error.to_string()))?,
    })
}

pub(crate) fn save_prompt_cache(
    args: &DecoderConfig,
    cache: &[Option<ConcatKeyValueCache>],
    destination: impl AsRef<Path>,
    descriptor: PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    options: &PromptCacheOptions,
    stream: &Stream,
) -> Result<PromptCacheManifest, Exception> {
    let layer_count = usize::try_from(args.num_hidden_layers)
        .map_err(|_| Exception::custom("invalid Muse-Glimmer cache layer count"))?;
    if cache.len() != layer_count {
        return Err(Exception::custom(format!(
            "Muse-Glimmer cache has {} layers, expected {layer_count}",
            cache.len()
        )));
    }
    let end = i64::try_from(prefix_token_ids.len())
        .map_err(|_| Exception::custom("Muse-Glimmer prompt length exceeds i64"))?;
    let mut blocks = Vec::with_capacity(layer_count);
    for (layer, cache) in cache.iter().enumerate() {
        let cache = cache.as_ref().ok_or_else(|| {
            Exception::custom(format!("Muse-Glimmer layer {layer} cache is missing"))
        })?;
        if i64::from(cache.offset()) != end {
            return Err(Exception::custom(format!(
                "Muse-Glimmer layer {layer} cache offset does not match the persisted prefix"
            )));
        }
        let (keys, values) = cache.snapshot_arrays(stream)?.ok_or_else(|| {
            Exception::custom(format!("Muse-Glimmer layer {layer} cache state is missing"))
        })?;
        let retained = i64::from(keys.dim(-2));
        blocks.push(PromptCacheSnapshotBlock {
            global_layer: layer,
            start: end.checked_sub(retained).ok_or_else(|| {
                Exception::custom(format!(
                    "Muse-Glimmer layer {layer} retained range underflow"
                ))
            })?,
            end,
            rank: None,
            arrays: CacheBlockArrays::KeyValue { keys, values },
        });
    }
    save_prompt_cache_snapshot(
        destination,
        descriptor,
        prefix_token_ids,
        blocks,
        &[],
        options,
    )
    .map_err(|error| Exception::custom(error.to_string()))
}

#[cfg(test)]
pub(crate) fn load_prompt_cache(
    args: &DecoderConfig,
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    stream: &Stream,
) -> Result<(Vec<Option<ConcatKeyValueCache>>, PromptCacheManifest), Exception> {
    let identity = prompt_cache_model_identity(args)?;
    load_prompt_cache_with_identity(
        args,
        directory,
        expected,
        prefix_token_ids,
        &identity,
        stream,
    )
}

pub(crate) fn load_prompt_cache_with_identity(
    args: &DecoderConfig,
    directory: impl AsRef<Path>,
    expected: &PromptCacheDescriptor,
    prefix_token_ids: &[u32],
    identity: &PromptCacheModelIdentity,
    stream: &Stream,
) -> Result<(Vec<Option<ConcatKeyValueCache>>, PromptCacheManifest), Exception> {
    let (blocks, state, manifest) =
        open_prompt_cache_snapshot(directory, expected, identity, prefix_token_ids, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
    if !state.is_empty() {
        return Err(Exception::custom(
            "Muse-Glimmer prompt cache contains unexpected fixed state",
        ));
    }
    let mut blocks = blocks
        .into_iter()
        .map(|block| (block.global_layer, block))
        .collect::<BTreeMap<_, _>>();
    let end = i32::try_from(prefix_token_ids.len())
        .map_err(|_| Exception::custom("Muse-Glimmer prompt length exceeds i32"))?;
    let mut cache = Vec::with_capacity(identity.layer_count);
    for layer in 0..identity.layer_count {
        let mut layer_cache = match args
            .attention_schedule
            .get(layer)
            .and_then(|policy| policy.window())
            .map(|window| {
                i32::try_from(window.get()).expect("validated Muse-Glimmer window fits i32")
            }) {
            Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
            None => ConcatKeyValueCache::new(),
        };
        let block = blocks.remove(&layer).ok_or_else(|| {
            Exception::custom(format!(
                "Muse-Glimmer layer {layer} prompt-cache block is missing"
            ))
        })?;
        match block.arrays {
            CacheBlockArrays::KeyValue { keys, values } => {
                layer_cache.restore_resident(keys, values, end)?;
            }
            CacheBlockArrays::CompressedLatentRotary { .. } => {
                return Err(Exception::custom(format!(
                    "Muse-Glimmer layer {layer} prompt cache contains compressed latent state"
                )))
            }
        }
        cache.push(Some(layer_cache));
    }
    if !blocks.is_empty() {
        return Err(Exception::custom(
            "Muse-Glimmer prompt cache contains unexpected attention blocks",
        ));
    }
    Ok((cache, manifest))
}

fn quantization_for(
    args: &DecoderConfig,
    prefix: Option<&str>,
    parameter: &str,
) -> Option<WeightQuantization> {
    match prefix {
        Some(prefix) => args.weight_quantization_for(&format!("{prefix}.{parameter}.weight")),
        None => args.weight_quantization(),
    }
}

pub(crate) fn rms_norm_without_scale(
    x: &Array,
    eps: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let dtype = x.dtype();
    let variance = mean_axis(&x.square(stream)?, -1, true, stream)?;
    x.multiply(
        rsqrt(variance.add(Array::from_f32(eps), stream)?, stream)?,
        stream,
    )?
    // The release implementation accumulates RMSNorm in float32, then
    // returns to the input dtype. Keeping the promoted value here turns every
    // following BF16 projection into an FP32 GEMV/GEMM.
    .as_dtype(dtype, stream)
}

fn forward_layer_norm(
    norm: &mut nn::RmsNorm,
    centered: bool,
    x: &Array,
    stream: &Stream,
) -> Result<Array, Exception> {
    let output = if centered {
        let scale = norm.weight.as_ref().add(Array::from_f32(1.0), stream)?;
        safemlx::fast::rms_norm(x, &scale, norm.eps, stream)?
    } else {
        norm.forward(x, stream)?
    };
    // Transformers' MuseGlimmerTextCenteredRMSNorm explicitly applies
    // type_as(x) after its float32 norm/scale calculation.
    output.as_dtype(x.dtype(), stream)
}

pub(crate) fn scale_logits(
    logits: Array,
    multiplier: f32,
    softcap: f32,
    stream: &Stream,
) -> Result<Array, Exception> {
    let scaled = logits.multiply(Array::from_f32(multiplier / softcap), stream)?;
    tanh(&scaled, stream)?.multiply(Array::from_f32(softcap), stream)
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Shared Muse-Glimmer attention layer.
pub struct Attention {
    /// Number of query heads.
    pub n_heads: i32,
    /// Number of key/value heads.
    pub n_kv_heads: i32,
    /// Attention scaling factor.
    pub scale: f32,

    #[quantizable]
    #[param]
    /// Query projection.
    pub q_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Key projection.
    pub k_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Value projection.
    pub v_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Output projection.
    pub o_proj: MaybeQuantized<nn::Linear>,
    #[quantizable]
    #[param]
    /// Per-head sigmoid gate applied before the output projection.
    pub gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    /// Query normalization.
    pub q_norm: Option<nn::RmsNorm>,
    #[param]
    /// Key normalization.
    pub k_norm: Option<nn::RmsNorm>,
    #[param]
    /// Rotary position embedding module.
    pub rope: RopeVariant,
    /// Layer-local attention window; absent for full attention.
    pub sliding_window: Option<i32>,
    /// Whether this layer applies rotary position embeddings.
    pub uses_rope: bool,
    /// Query scale applied after weightless Q/K normalization.
    pub qk_scale_factor: f32,
    /// Epsilon for weightless query/key normalization.
    pub qk_norm_eps: f32,
}

impl Attention {
    pub(crate) fn new_for_layer(
        args: &DecoderConfig,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer = usize::try_from(layer_index)
            .map_err(|_| Exception::custom(format!("invalid Qwen layer index {layer_index}")))?;
        let policy = args.attention_schedule.get(layer).ok_or_else(|| {
            Exception::custom(format!(
                "Qwen attention schedule has no policy for layer {layer_index}"
            ))
        })?;
        Self::new_with_prefix(
            args,
            &format!("model.layers.{layer_index}.self_attn"),
            *policy,
            *args.layer_uses_rope.get(layer).ok_or_else(|| {
                Exception::custom(format!(
                    "Muse-Glimmer RoPE schedule has no layer {layer_index}"
                ))
            })?,
            stream,
        )
    }

    fn new_with_prefix(
        args: &DecoderConfig,
        prefix: &str,
        policy: AttentionPolicy,
        uses_rope: bool,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let dim = args.hidden_size;
        let n_heads = args.num_attention_heads;
        let n_kv_heads = args.num_key_value_heads;

        let head_dim = args.head_dim;
        let scale = (head_dim as f32).sqrt().recip();

        let q_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_heads * head_dim,
            args.qkv_bias(),
            quantization_for(args, Some(prefix), "q_proj"),
            stream,
        )?;
        let k_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.qkv_bias(),
            quantization_for(args, Some(prefix), "k_proj"),
            stream,
        )?;
        let v_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.qkv_bias(),
            quantization_for(args, Some(prefix), "v_proj"),
            stream,
        )?;
        let o_proj = common::linear::unloaded_maybe_quantized_linear(
            n_heads * head_dim,
            dim,
            false,
            quantization_for(args, Some(prefix), "o_proj"),
            stream,
        )?;
        let gate_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_heads * head_dim,
            false,
            quantization_for(args, Some(prefix), "gate_proj"),
            stream,
        )?;

        // The official GGUF conversion synthesizes unit Q/K RMSNorm scales.
        // Safetensors remains weightless and must not invent parameters.
        let q_norm = matches!(args.weight_convention, WeightConvention::Gguf)
            .then(|| nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream))
            .transpose()?;
        let k_norm = matches!(args.weight_convention, WeightConvention::Gguf)
            .then(|| nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream))
            .transpose()?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            false,
            &args.rope_scaling,
            args.max_position_embeddings,
            stream,
        )?;

        Ok(Self {
            n_heads,
            n_kv_heads,
            scale,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            gate_proj,
            q_norm,
            k_norm,
            rope,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(|_| Exception::custom("sliding attention window exceeds i32"))?,
            uses_rope,
            qk_scale_factor: args.qk_scale_factor,
            qk_norm_eps: args.rms_norm_eps,
        })
    }

    fn normalize_query(&mut self, value: Array, stream: &Stream) -> Result<Array, Exception> {
        let dtype = value.dtype();
        let value = match self.q_norm.as_mut() {
            Some(norm) => norm.forward(&value, stream)?,
            None => rms_norm_without_scale(&value, self.qk_norm_eps, stream)?,
        };
        value
            .multiply(Array::from_f32(self.qk_scale_factor), stream)?
            .as_dtype(dtype, stream)
    }

    fn normalize_key(&mut self, value: Array, stream: &Stream) -> Result<Array, Exception> {
        match self.k_norm.as_mut() {
            Some(norm) => norm.forward(&value, stream),
            None => rms_norm_without_scale(&value, self.qk_norm_eps, stream),
        }
    }

    fn gate_output(
        &mut self,
        input: &Array,
        attended: Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let gate = sigmoid(&self.gate_proj.forward(input, stream)?, stream)?;
        self.o_proj
            .forward(&attended.multiply(gate, stream)?, stream)
    }

    fn apply_positions<C: KeyValueCache>(
        &mut self,
        queries: Array,
        keys: Array,
        values: Array,
        cache: &mut Option<&mut C>,
        stream: &Stream,
    ) -> Result<(Array, Array, Array), Exception> {
        if self.uses_rope {
            return apply_rope_and_update_cache(
                &mut self.rope,
                queries,
                keys,
                values,
                cache,
                stream,
            );
        }
        let (keys, values) = match cache.as_mut() {
            Some(cache) => cache.update_and_fetch(keys, values, stream)?,
            None => (keys, values),
        };
        Ok((queries, keys, values))
    }

    fn apply_explicit_positions<C: KeyValueCache>(
        &mut self,
        queries: Array,
        keys: Array,
        values: Array,
        cos: &Array,
        sin: &Array,
        cache: &mut Option<&mut C>,
        stream: &Stream,
    ) -> Result<(Array, Array, Array), Exception> {
        if self.uses_rope {
            return apply_rotary_embeddings_and_update_cache(
                queries, keys, values, cos, sin, cache, stream,
            );
        }
        let (keys, values) = match cache.as_mut() {
            Some(cache) => cache.update_and_fetch(keys, values, stream)?,
            None => (keys, values),
        };
        Ok((queries, keys, values))
    }

    /// Forward pass that reports attention activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, mut cache } = input;

        let (batch, seq_len) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &queries)?;
        let keys = self.k_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.k_proj"), &keys)?;
        let values = self.v_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.v_proj"), &values)?;

        let queries = self.normalize_query(
            reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.q_norm"), &queries)?;
        let keys = self.normalize_key(
            reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?,
            stream,
        )?;
        observer.observe(&format!("{prefix}.k_norm"), &keys)?;
        let values = reshape_attention_projection(values, batch, seq_len, self.n_kv_heads, stream)?;
        observer.observe(&format!("{prefix}.values"), &values)?;

        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            self.apply_positions(queries, keys, values, &mut cache, stream)?;
        observer.observe(&format!("{prefix}.queries_rope"), &queries)?;
        observer.observe(&format!("{prefix}.keys_rope"), &keys)?;
        observer.observe(&format!("{prefix}.values_cache"), &values)?;
        let output = if let Some(window) = self.sliding_window.filter(|_| seq_len > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                window,
                position_offset,
                batch,
                seq_len,
                stream,
            )?
        } else {
            let attention_probs =
                attention_probabilities(&queries, &keys, self.scale, mask, stream)?;
            observer.observe(&format!("{prefix}.attention_probs"), &attention_probs)?;
            finish_attention(
                queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
            )?
        };
        observer.observe(&format!("{prefix}.attention"), &output)?;

        let gate = sigmoid(&self.gate_proj.forward(x, stream)?, stream)?;
        observer.observe(&format!("{prefix}.gate"), &gate)?;
        let output = self
            .o_proj
            .forward(&output.multiply(gate, stream)?, stream)?;
        observer.observe(&format!("{prefix}.o_proj"), &output)?;
        Ok(output)
    }

    pub(crate) fn forward_with_rotary_embeddings<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, mut cache } = input;
        let (batch, seq_len) = batch_seq(x);
        let queries = self.q_proj.forward(x, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let queries = self.normalize_query(
            reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?,
            stream,
        )?;
        let keys = self.normalize_key(
            reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?,
            stream,
        )?;
        let values = reshape_attention_projection(
            self.v_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            self.apply_explicit_positions(queries, keys, values, cos, sin, &mut cache, stream)?;
        let output = if let Some(window) = self.sliding_window.filter(|_| seq_len > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                window,
                position_offset,
                batch,
                seq_len,
                stream,
            )?
        } else {
            finish_attention(
                queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
            )?
        };
        self.gate_output(x, output, stream)
    }

    pub(crate) fn forward_with_rotary_embeddings_tensor_parallel<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, mut cache } = input;
        let (batch, seq_len) = batch_seq(x);
        let queries = self.q_proj.forward(x, stream)?;
        let queries = reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?;
        let queries = self.normalize_query(queries, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let keys = reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?;
        let keys = self.normalize_key(keys, stream)?;
        let values = reshape_attention_projection(
            self.v_proj.forward(x, stream)?,
            batch,
            seq_len,
            self.n_kv_heads,
            stream,
        )?;
        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            self.apply_explicit_positions(queries, keys, values, cos, sin, &mut cache, stream)?;
        let attended = if let Some(window) = self.sliding_window.filter(|_| seq_len > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                window,
                position_offset,
                batch,
                seq_len,
                stream,
            )?
        } else {
            finish_attention(
                queries, keys, values, cache, self.scale, mask, batch, seq_len, stream,
            )?
        };
        let gate = sigmoid(&self.gate_proj.forward(x, stream)?, stream)?;
        crate::nn::parallel::forward_row_parallel(
            &mut self.o_proj,
            &attended.multiply(gate, stream)?,
            group,
            stream,
        )
    }
}

impl<C> Module<AttentionInput<'_, C>> for Attention
where
    C: KeyValueCache,
{
    type Output = Array;

    type Error = Exception;

    #[allow(non_snake_case)]
    fn forward(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, mut cache } = input;

        let (B, L) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let values = self.v_proj.forward(x, stream)?;

        let queries = self.normalize_query(
            reshape_attention_projection(queries, B, L, self.n_heads, stream)?,
            stream,
        )?;
        let keys = self.normalize_key(
            reshape_attention_projection(keys, B, L, self.n_kv_heads, stream)?,
            stream,
        )?;
        let values = reshape_attention_projection(values, B, L, self.n_kv_heads, stream)?;
        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            self.apply_positions(queries, keys, values, &mut cache, stream)?;
        let output = if let Some(window) = self.sliding_window.filter(|_| L > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                window,
                position_offset,
                B,
                L,
                stream,
            )?
        } else {
            finish_attention(queries, keys, values, cache, self.scale, mask, B, L, stream)?
        };

        self.gate_output(x, output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
        self.gate_proj.training_mode(mode);
        if let Some(norm) = &mut self.q_norm {
            norm.training_mode(mode);
        }
        if let Some(norm) = &mut self.k_norm {
            norm.training_mode(mode);
        }
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Dense-Qwen SwiGLU feed-forward block.
pub type Mlp = SwiGluMlp;

/// Packed routed-expert bank shared with other SwiGLU MoE architectures.
pub type Experts = common::moe::PackedSwiGluExperts;

#[derive(Debug, Clone, ModuleParameters)]
/// Qwen3 sparse MoE feed-forward block.
pub struct SparseMoeBlock {
    #[param]
    /// Top-k router.
    pub gate: common::moe::TopKRouter,
    #[param]
    /// Routed expert bank.
    pub experts: Experts,
}

impl SparseMoeBlock {
    fn new(args: &DecoderConfig, layer_index: i32, stream: &Stream) -> Result<Self, Exception> {
        Ok(Self {
            gate: common::moe::TopKRouter::new(
                common::moe::TopKRouterConfig {
                    top_k: args.num_experts_per_tok,
                    num_experts: args.num_experts,
                    hidden_size: args.hidden_size,
                    score_function: TopKRouterScoreFunction::Softmax,
                    norm_topk_prob: args.norm_topk_prob,
                    normalization_epsilon: 0.0,
                    routed_scaling_factor: 1.0,
                    n_group: 1,
                    topk_group: 1,
                    score_correction_bias: false,
                },
                stream,
            )?,
            experts: Experts::new(
                args.num_experts,
                args.hidden_size,
                args.moe_intermediate_size,
                args.weight_quantization_for(&format!(
                    "model.layers.{layer_index}.mlp.experts.gate_up_proj"
                )),
                args.weight_quantization_for(&format!(
                    "model.layers.{layer_index}.mlp.experts.down_proj"
                )),
                stream,
            )?,
        })
    }

    pub(crate) fn forward_tensor_parallel(
        &mut self,
        hidden_states: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        let partial = self.experts.forward(&flat, &indices, &weights, stream)?;
        safemlx::distributed::all_sum(&partial, group, stream)?.reshape(shape, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        let returned = crate::architectures::distributed::expert::dispatch_replicated(
            &flat,
            &indices,
            &weights,
            assignment,
            &mut self.experts,
            expert_group,
            stream,
        )
        .map_err(|error| Exception::custom(error.to_string()))?;
        statistics.accumulate(&returned.statistics);
        safemlx::distributed::all_sum(&returned.reduced_output, tensor_group, stream)?
            .reshape(shape, stream)
    }

    fn forward_with_observer(
        &mut self,
        hidden_states: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let routing =
            self.gate
                .forward_with_observer(&flat, stream, &format!("{prefix}.gate"), observer)?;
        let output = self
            .experts
            .forward(&flat, &routing.indices, &routing.weights, stream)?;
        observer.observe(&format!("{prefix}.experts.output"), &output)?;
        observer.observe_moe_routing(MoeRoutingObservation {
            prefix,
            selected_experts: &routing.indices,
            selected_scores: &routing.scores,
            routing_weights: &routing.weights,
            routed_output: &output,
            local_routed_output: None,
            reduced_routed_output: Some(&output),
            shared_output: None,
            combined_output: Some(&output),
            num_experts: self.gate.num_experts,
        })?;
        output.reshape(shape, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        mut observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&flat])?;
        let moe_started = std::time::Instant::now();
        let previous_moe_time = statistics.total_time;
        let router_started = std::time::Instant::now();
        let (indices, selected_scores, weights) = if let Some(observer) = observer.as_deref_mut() {
            let routing = self.gate.forward_with_observer(
                &flat,
                stream,
                &format!("{prefix}.gate"),
                observer,
            )?;
            (routing.indices, Some(routing.scores), routing.weights)
        } else {
            let (indices, weights) = self.gate.forward(&flat, stream)?;
            (indices, None, weights)
        };
        let mut router_outputs = vec![&indices, &weights];
        if let Some(scores) = selected_scores.as_ref() {
            router_outputs.push(scores);
        }
        crate::architectures::distributed::expert::materialize_timing_phase(router_outputs)?;
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
        if let Some(observer) = observer {
            observer.observe(
                &format!("{prefix}.experts.local_output"),
                &returned.local_output,
            )?;
            observer.observe(
                &format!("{prefix}.experts.reduced_output"),
                &returned.reduced_output,
            )?;
            observer.observe_moe_routing(MoeRoutingObservation {
                prefix,
                selected_experts: &indices,
                selected_scores: selected_scores
                    .as_ref()
                    .expect("observed EP routing scores initialized"),
                routing_weights: &weights,
                routed_output: &returned.reduced_output,
                local_routed_output: Some(&returned.local_output),
                reduced_routed_output: Some(&returned.reduced_output),
                shared_output: None,
                combined_output: Some(&returned.reduced_output),
                num_experts: self.gate.num_experts,
            })?;
        }
        let output = returned.reduced_output.reshape(shape, stream)?;
        crate::architectures::distributed::expert::materialize_timing_phase([&output])?;
        statistics.total_time = previous_moe_time + moe_started.elapsed();
        Ok(output)
    }
}

impl Module<&Array> for SparseMoeBlock {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Array, Exception> {
        let shape = input.shape();
        let flat = input.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        self.experts
            .forward(&flat, &indices, &weights, stream)?
            .reshape(shape, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.gate.training_mode(mode);
        self.experts.training_mode(mode);
    }
}

#[derive(Debug, Clone)]
/// Dense or sparse Qwen3 feed-forward layer stored under the checkpoint-native `mlp` namespace.
pub enum FeedForward {
    /// Dense SwiGLU MLP.
    Dense(Mlp),
    /// Sparse mixture-of-experts block.
    Moe(SparseMoeBlock),
}

impl FeedForward {
    fn new(args: &DecoderConfig, layer_index: i32, stream: &Stream) -> Result<Self, Exception> {
        if args.is_moe() {
            Ok(Self::Moe(SparseMoeBlock::new(args, layer_index, stream)?))
        } else {
            let prefix = format!("model.layers.{layer_index}.mlp");
            Ok(Self::Dense(SwiGluMlp {
                gate_proj: common::linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.intermediate_size,
                    false,
                    args.weight_quantization_for(&format!("{prefix}.gate_proj.weight")),
                    stream,
                )?,
                down_proj: common::linear::unloaded_maybe_quantized_linear(
                    args.intermediate_size,
                    args.hidden_size,
                    false,
                    args.weight_quantization_for(&format!("{prefix}.down_proj.weight")),
                    stream,
                )?,
                up_proj: common::linear::unloaded_maybe_quantized_linear(
                    args.hidden_size,
                    args.intermediate_size,
                    false,
                    args.weight_quantization_for(&format!("{prefix}.up_proj.weight")),
                    stream,
                )?,
            }))
        }
    }

    fn is_moe(&self) -> bool {
        matches!(self, Self::Moe(_))
    }

    fn forward_with_observer(
        &mut self,
        input: &Array,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward_with_observer(input, stream, prefix, observer),
            Self::Moe(moe) => moe.forward_with_observer(input, stream, prefix, observer),
        }
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel(
        &mut self,
        hidden_states: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        match self {
            Self::Dense(mlp) => mlp.forward(hidden_states, stream),
            Self::Moe(moe) => moe.forward_expert_parallel(
                hidden_states,
                assignment,
                group,
                statistics,
                prefix,
                observer,
                stream,
            ),
        }
    }
}

impl Module<&Array> for FeedForward {
    type Output = Array;
    type Error = Exception;

    fn forward(&mut self, input: &Array, stream: &Stream) -> Result<Self::Output, Self::Error> {
        match self {
            Self::Dense(mlp) => mlp.forward(input, stream),
            Self::Moe(moe) => moe.forward(input, stream),
        }
    }

    fn training_mode(&mut self, mode: bool) {
        match self {
            Self::Dense(mlp) => mlp.training_mode(mode),
            Self::Moe(moe) => moe.training_mode(mode),
        }
    }
}

impl ModuleParametersTrait for FeedForward {
    fn num_parameters(&self) -> usize {
        match self {
            Self::Dense(mlp) => mlp.num_parameters(),
            Self::Moe(moe) => moe.num_parameters(),
        }
    }

    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters(),
            Self::Moe(moe) => moe.parameters(),
        }
    }

    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        match self {
            Self::Dense(mlp) => mlp.parameters_mut(),
            Self::Moe(moe) => moe.parameters_mut(),
        }
    }

    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        match self {
            Self::Dense(mlp) => mlp.trainable_parameters(),
            Self::Moe(moe) => moe.trainable_parameters(),
        }
    }

    fn freeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.freeze_parameters(recursive),
            Self::Moe(moe) => moe.freeze_parameters(recursive),
        }
    }

    fn unfreeze_parameters(&mut self, recursive: bool) {
        match self {
            Self::Dense(mlp) => mlp.unfreeze_parameters(recursive),
            Self::Moe(moe) => moe.unfreeze_parameters(recursive),
        }
    }

    fn all_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.all_frozen(),
            Self::Moe(moe) => moe.all_frozen(),
        }
    }

    fn any_frozen(&self) -> Option<bool> {
        match self {
            Self::Dense(mlp) => mlp.any_frozen(),
            Self::Moe(moe) => moe.any_frozen(),
        }
    }
}

impl safemlx::quantization::Quantizable for FeedForward {
    type Quantized = Self;
    type QuantizationError = Exception;

    fn try_into_quantized(
        self,
        group_size: i32,
        bits: i32,
        stream: &Stream,
    ) -> Result<Self::Quantized, Self::QuantizationError> {
        match self {
            Self::Dense(mlp) => Ok(Self::Dense(
                safemlx::quantization::Quantizable::try_into_quantized(
                    mlp, group_size, bits, stream,
                )?,
            )),
            Self::Moe(moe) => Ok(Self::Moe(moe)),
        }
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Shared Muse-Glimmer decoder block.
pub struct TransformerBlock {
    /// Number of attention heads.
    pub num_attention_heads: i32,
    /// Transformer hidden size.
    pub hidden_size: i32,

    #[quantizable]
    #[param]
    /// Self-attention layer.
    pub self_attn: Attention,

    #[quantizable]
    #[param]
    /// Dense or sparse feed-forward layer.
    pub mlp: FeedForward,

    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,

    #[param]
    /// Post-attention RMSNorm applied before the residual addition.
    pub post_attention_layernorm: nn::RmsNorm,
    #[param]
    /// Pre-feed-forward RMSNorm.
    pub pre_feedforward_layernorm: nn::RmsNorm,
    #[param]
    /// Post-feed-forward RMSNorm applied before the residual addition.
    pub post_feedforward_layernorm: nn::RmsNorm,
    /// Whether checkpoint layer-norm weights are centered around zero.
    pub centered_norm_weights: bool,
}

impl TransformerBlock {
    /// Creates an unloaded decoder block from model arguments.
    pub fn new(args: &DecoderConfig, stream: &Stream) -> Result<Self, Exception> {
        Self::new_for_layer(args, 0, stream)
    }

    pub(crate) fn new_for_layer(
        args: &DecoderConfig,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new_for_layer(args, layer_index, stream)?;
        let mlp = FeedForward::new(args, layer_index, stream)?;
        let input_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let post_attention_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.post_norm_eps, Dtype::Float32, stream)?;
        let pre_feedforward_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;
        let post_feedforward_layernorm =
            nn::RmsNorm::unloaded(args.hidden_size, args.post_norm_eps, Dtype::Float32, stream)?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            centered_norm_weights: args.weight_convention == WeightConvention::HuggingFace,
        })
    }

    fn normalize_input(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        forward_layer_norm(
            &mut self.input_layernorm,
            self.centered_norm_weights,
            x,
            stream,
        )
    }

    fn normalize_post_attention(&mut self, x: &Array, stream: &Stream) -> Result<Array, Exception> {
        forward_layer_norm(
            &mut self.post_attention_layernorm,
            self.centered_norm_weights,
            x,
            stream,
        )
    }

    fn normalize_pre_feedforward(
        &mut self,
        x: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        forward_layer_norm(
            &mut self.pre_feedforward_layernorm,
            self.centered_norm_weights,
            x,
            stream,
        )
    }

    fn normalize_post_feedforward(
        &mut self,
        x: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        forward_layer_norm(
            &mut self.post_feedforward_layernorm,
            self.centered_norm_weights,
            x,
            stream,
        )
    }

    /// Forward pass that reports block activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
        prefix: &str,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;

        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let normed = self.normalize_input(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &normed)?;

        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let r = self.self_attn.forward_with_observer(
            self_attn_input,
            stream,
            &format!("{prefix}.self_attn"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.self_attn_output"), &r)?;
        let r = self.normalize_post_attention(&r, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_attention"), &r)?;
        let h = x.add(r, stream)?;
        observer.observe(&format!("{prefix}.post_attention_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_attention"), &h)?;

        let feed_forward_name = if self.mlp.is_moe() { "moe" } else { "mlp" };
        observer.observe(&format!("{prefix}.residual_before_{feed_forward_name}"), &h)?;
        let post_normed = self.normalize_pre_feedforward(&h, stream)?;
        observer.observe(&format!("{prefix}.pre_feedforward_layernorm"), &post_normed)?;
        let r = self.mlp.forward_with_observer(
            &post_normed,
            stream,
            &format!("{prefix}.mlp"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.{feed_forward_name}_output"), &r)?;
        let r = self.normalize_post_feedforward(&r, stream)?;
        observer.observe(&format!("{prefix}.post_feedforward_layernorm"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_{feed_forward_name}"), &r)?;
        let output = h.add(r, stream)?;
        let output = observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output);
        observer.observe(&format!("{prefix}.output"), &output)?;
        observer.observe(
            &format!("{prefix}.residual_after_{feed_forward_name}"),
            &output,
        )?;
        Ok(output)
    }

    pub(crate) fn forward_with_rotary_embeddings<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.normalize_input(x, stream)?;
        let attention = self.self_attn.forward_with_rotary_embeddings(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            cos,
            sin,
            stream,
        )?;
        let attention = self.normalize_post_attention(&attention, stream)?;
        let hidden = x.add(attention, stream)?;
        let normed = self.normalize_pre_feedforward(&hidden, stream)?;
        let mlp = self.mlp.forward(&normed, stream)?;
        let mlp = self.normalize_post_feedforward(&mlp, stream)?;
        hidden.add(mlp, stream)
    }

    /// Executes a block whose attention heads and dense/expert intermediates
    /// are rank-local, reducing row projections exactly once.
    pub(crate) fn forward_tensor_parallel<C: KeyValueCache>(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.normalize_input(hidden, stream)?;
        let attention = &mut self.self_attn;
        let (batch, sequence) = batch_seq(&normalized);
        let queries = attention.q_proj.forward(&normalized, stream)?;
        let keys = attention.k_proj.forward(&normalized, stream)?;
        let values = attention.v_proj.forward(&normalized, stream)?;
        let queries =
            reshape_attention_projection(queries, batch, sequence, attention.n_heads, stream)?;
        let queries = attention.normalize_query(queries, stream)?;
        let keys =
            reshape_attention_projection(keys, batch, sequence, attention.n_kv_heads, stream)?;
        let keys = attention.normalize_key(keys, stream)?;
        let values =
            reshape_attention_projection(values, batch, sequence, attention.n_kv_heads, stream)?;
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let mut cache = cache;
        let (queries, keys, values) =
            attention.apply_positions(queries, keys, values, &mut cache, stream)?;
        let attended = if let Some(window) = attention.sliding_window.filter(|_| sequence > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                attention.scale,
                window,
                offset,
                batch,
                sequence,
                stream,
            )?
        } else {
            finish_attention(
                queries,
                keys,
                values,
                cache,
                attention.scale,
                mask,
                batch,
                sequence,
                stream,
            )?
        };
        let gate = sigmoid(&attention.gate_proj.forward(&normalized, stream)?, stream)?;
        let attention = crate::nn::parallel::forward_row_parallel(
            &mut attention.o_proj,
            &attended.multiply(gate, stream)?,
            group,
            stream,
        )?;
        let attention = self.normalize_post_attention(&attention, stream)?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.normalize_pre_feedforward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate =
                    crate::nn::layers::silu(mlp.gate_proj.forward(&normalized, stream)?, stream)?;
                let up = mlp.up_proj.forward(&normalized, stream)?;
                crate::nn::parallel::forward_row_parallel(
                    &mut mlp.down_proj,
                    &gate.multiply(up, stream)?,
                    group,
                    stream,
                )?
            }
            FeedForward::Moe(moe) => moe.forward_tensor_parallel(&normalized, group, stream)?,
        };
        let feed_forward = self.normalize_post_feedforward(&feed_forward, stream)?;
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded attention and expert intermediates while routing
    /// only through the matching stage/TP-coordinate EP group.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel<C: KeyValueCache>(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let attention = &mut self.self_attn;
        let (batch, sequence) = batch_seq(&normalized);
        let queries = attention.q_proj.forward(&normalized, stream)?;
        let keys = attention.k_proj.forward(&normalized, stream)?;
        let values = attention.v_proj.forward(&normalized, stream)?;
        let queries =
            reshape_attention_projection(queries, batch, sequence, attention.n_heads, stream)?;
        let queries = match attention.q_norm.as_mut() {
            Some(norm) => norm.forward(&queries, stream)?,
            None => queries,
        };
        let keys =
            reshape_attention_projection(keys, batch, sequence, attention.n_kv_heads, stream)?;
        let keys = match attention.k_norm.as_mut() {
            Some(norm) => norm.forward(&keys, stream)?,
            None => keys,
        };
        let values =
            reshape_attention_projection(values, batch, sequence, attention.n_kv_heads, stream)?;
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let mut cache = cache;
        let (queries, keys, values) = apply_rope_and_update_cache(
            &mut attention.rope,
            queries,
            keys,
            values,
            &mut cache,
            stream,
        )?;
        let attended = if let Some(window) = attention.sliding_window.filter(|_| sequence > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                attention.scale,
                window,
                offset,
                batch,
                sequence,
                stream,
            )?
        } else {
            finish_attention(
                queries,
                keys,
                values,
                cache,
                attention.scale,
                mask,
                batch,
                sequence,
                stream,
            )?
        };
        let attention = crate::nn::parallel::forward_row_parallel(
            &mut attention.o_proj,
            &attended,
            tensor_group,
            stream,
        )?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let FeedForward::Moe(moe) = &mut self.mlp else {
            return Err(Exception::custom(
                "tensor+expert execution requires a Qwen3-MoE layer",
            ));
        };
        let feed_forward = moe.forward_tensor_expert_parallel(
            &normalized,
            assignment,
            tensor_group,
            expert_group,
            statistics,
            stream,
        )?;
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded MRoPE attention and EP-local routed experts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_tensor_expert_parallel_with_rotary<C: KeyValueCache>(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        cos: &Array,
        sin: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let normalized = self.input_layernorm.forward(hidden, stream)?;
        let attention = self
            .self_attn
            .forward_with_rotary_embeddings_tensor_parallel(
                AttentionInput {
                    x: &normalized,
                    mask,
                    cache,
                },
                cos,
                sin,
                tensor_group,
                stream,
            )?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let FeedForward::Moe(moe) = &mut self.mlp else {
            return Err(Exception::custom(
                "tensor+expert MRoPE execution requires a Qwen3-MoE layer",
            ));
        };
        let feed_forward = moe.forward_tensor_expert_parallel(
            &normalized,
            assignment,
            tensor_group,
            expert_group,
            statistics,
            stream,
        )?;
        hidden.add(feed_forward, stream)
    }

    pub(crate) fn forward_with_rotary_embeddings_tensor_parallel<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.normalize_input(x, stream)?;
        let attention = self
            .self_attn
            .forward_with_rotary_embeddings_tensor_parallel(
                AttentionInput {
                    x: &normed,
                    mask,
                    cache,
                },
                cos,
                sin,
                group,
                stream,
            )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate =
                    crate::nn::layers::silu(mlp.gate_proj.forward(&normed, stream)?, stream)?;
                let up = mlp.up_proj.forward(&normed, stream)?;
                crate::nn::parallel::forward_row_parallel(
                    &mut mlp.down_proj,
                    &gate.multiply(up, stream)?,
                    group,
                    stream,
                )?
            }
            FeedForward::Moe(moe) => moe.forward_tensor_parallel(&normed, group, stream)?,
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes a block while delegating routed-expert evaluation to a compact bank.
    pub(crate) fn forward_sparse_experts<C, F>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normed, stream)?,
            FeedForward::Moe(moe) => {
                let shape = normed.shape();
                let flat = normed.reshape(&[-1, normed.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes TP-sharded attention and dense projections while delegating
    /// routed experts to the matching-coordinate EP exchange group.
    pub(crate) fn forward_sparse_experts_tensor_parallel<C, F>(
        &mut self,
        input: AttentionInput<'_, C>,
        tensor_group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = &mut self.self_attn;
        let (batch, sequence) = batch_seq(&normed);
        let queries = attention.q_proj.forward(&normed, stream)?;
        let keys = attention.k_proj.forward(&normed, stream)?;
        let values = attention.v_proj.forward(&normed, stream)?;
        let queries =
            reshape_attention_projection(queries, batch, sequence, attention.n_heads, stream)?;
        let queries = match attention.q_norm.as_mut() {
            Some(norm) => norm.forward(&queries, stream)?,
            None => queries,
        };
        let keys =
            reshape_attention_projection(keys, batch, sequence, attention.n_kv_heads, stream)?;
        let keys = match attention.k_norm.as_mut() {
            Some(norm) => norm.forward(&keys, stream)?,
            None => keys,
        };
        let values =
            reshape_attention_projection(values, batch, sequence, attention.n_kv_heads, stream)?;
        let offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let mut cache = cache;
        let (queries, keys, values) = apply_rope_and_update_cache(
            &mut attention.rope,
            queries,
            keys,
            values,
            &mut cache,
            stream,
        )?;
        let attended = if let Some(window) = attention.sliding_window.filter(|_| sequence > 1) {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                attention.scale,
                window,
                offset,
                batch,
                sequence,
                stream,
            )?
        } else {
            finish_attention(
                queries,
                keys,
                values,
                cache,
                attention.scale,
                mask,
                batch,
                sequence,
                stream,
            )?
        };
        let attention = crate::nn::parallel::forward_row_parallel(
            &mut attention.o_proj,
            &attended,
            tensor_group,
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate =
                    crate::nn::layers::silu(mlp.gate_proj.forward(&normed, stream)?, stream)?;
                let up = mlp.up_proj.forward(&normed, stream)?;
                crate::nn::parallel::forward_row_parallel(
                    &mut mlp.down_proj,
                    &gate.multiply(up, stream)?,
                    tensor_group,
                    stream,
                )?
            }
            FeedForward::Moe(moe) => {
                let shape = normed.shape();
                let flat = normed.reshape(&[-1, normed.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    pub(crate) fn forward_sparse_experts_with_rotary<C, F>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward_with_rotary_embeddings(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            cos,
            sin,
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normed, stream)?,
            FeedForward::Moe(moe) => {
                let shape = normed.shape();
                let flat = normed.reshape(&[-1, normed.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    /// Executes MRoPE attention and dense projections over the local TP shard
    /// while delegating routed experts to the matching-coordinate EP group.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_sparse_experts_with_rotary_tensor_parallel<C, F>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        tensor_group: &safemlx::distributed::Group,
        stream: &Stream,
        execute: F,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
        F: FnOnce(&Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self
            .self_attn
            .forward_with_rotary_embeddings_tensor_parallel(
                AttentionInput {
                    x: &normed,
                    mask,
                    cache,
                },
                cos,
                sin,
                tensor_group,
                stream,
            )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate =
                    crate::nn::layers::silu(mlp.gate_proj.forward(&normed, stream)?, stream)?;
                let up = mlp.up_proj.forward(&normed, stream)?;
                crate::nn::parallel::forward_row_parallel(
                    &mut mlp.down_proj,
                    &gate.multiply(up, stream)?,
                    tensor_group,
                    stream,
                )?
            }
            FeedForward::Moe(moe) => {
                let shape = normed.shape();
                let flat = normed.reshape(&[-1, normed.dim(-1)], stream)?;
                let (indices, weights) = moe.gate.forward(&flat, stream)?;
                execute(&flat, &indices, &weights, stream)?.reshape(shape, stream)?
            }
        };
        hidden.add(feed_forward, stream)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = self.mlp.forward_expert_parallel(
            &normed,
            assignment,
            group,
            statistics,
            &format!("{prefix}.mlp"),
            observer,
            stream,
        )?;
        hidden.add(mlp, stream)
    }

    /// Runs explicit externally supplied rotary embeddings with an EP-local
    /// routed expert bank. Multimodal Qwen stages use this to keep MRoPE
    /// semantics independent from the pipeline and exchange runtimes.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn forward_expert_parallel_with_rotary<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        cos: &Array,
        sin: &Array,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let AttentionInput { x, mask, cache } = input;
        let normed = self.input_layernorm.forward(x, stream)?;
        let attention = self.self_attn.forward_with_rotary_embeddings(
            AttentionInput {
                x: &normed,
                mask,
                cache,
            },
            cos,
            sin,
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let mlp = self.mlp.forward_expert_parallel(
            &normed,
            assignment,
            group,
            statistics,
            &format!("{prefix}.mlp"),
            observer,
            stream,
        )?;
        hidden.add(mlp, stream)
    }
}

impl<C> Module<AttentionInput<'_, C>> for TransformerBlock
where
    C: KeyValueCache,
{
    type Output = Array;

    type Error = Exception;

    fn forward(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let AttentionInput { x, mask, cache } = input;

        let normed = self.normalize_input(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let r = self.self_attn.forward(self_attn_input, stream)?;
        let r = self.normalize_post_attention(&r, stream)?;
        let h = x.add(r, stream)?;

        let post_normed = self.normalize_pre_feedforward(&h, stream)?;
        let r = self.mlp.forward(&post_normed, stream)?;
        let r = self.normalize_post_feedforward(&r, stream)?;
        h.add(r, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
        self.pre_feedforward_layernorm.training_mode(mode);
        self.post_feedforward_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Shared Muse-Glimmer transformer body without the language-model head.
pub struct Decoder {
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of decoder layers.
    pub num_hidden_layers: i32,

    #[quantizable]
    #[param]
    /// Token embedding table.
    pub embed_tokens: MaybeQuantized<nn::Embedding>,

    #[quantizable]
    #[param]
    /// Decoder blocks.
    pub layers: Vec<TransformerBlock>,

    #[param]
    /// Final RMSNorm.
    pub norm: nn::RmsNorm,
}

fn full_attention_mask<C: KeyValueCache>(
    hidden: &Array,
    cache: &[Option<C>],
    explicit_mask: bool,
    stream: &Stream,
) -> Result<Option<Array>, Exception> {
    if explicit_mask || hidden.dim(1) <= 1 {
        return Ok(None);
    }
    let offset = cache
        .first()
        .and_then(Option::as_ref)
        .map_or(0, KeyValueCache::offset);
    create_causal_mask(hidden.dim(1), Some(offset), None, None, stream).map(Some)
}

impl Decoder {
    /// Creates an unloaded Muse-Glimmer transformer body.
    pub fn new(args: &DecoderConfig, stream: &Stream) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());
        validate_attention_schedule(args).map_err(|error| Exception::custom(error.to_string()))?;

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.weight_quantization_for("model.embed_tokens.weight"),
            stream,
        )?;
        let layers = (0..num_hidden_layers)
            .map(|layer_index| TransformerBlock::new_for_layer(args, layer_index, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let norm =
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;

        Ok(Self {
            vocab_size,
            num_hidden_layers,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Forward pass that reports transformer-body activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs, stream)?;
        h = rms_norm_without_scale(&h, self.norm.eps, stream)?;
        observer.observe("model.embed_tokens", &h)?;

        let full_mask = full_attention_mask(&h, cache, mask.is_some(), stream)?;
        if let Some(mask) = mask.or(full_mask.as_ref()) {
            observer.observe("model.attention_mask", mask)?;
        }

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_mask = match mask {
                Some(mask) => Some(mask),
                None if layer.self_attn.sliding_window.is_none() => full_mask.as_ref(),
                None => None,
            };
            let layer_input = AttentionInput {
                x: &h,
                mask: layer_mask,
                cache: c.as_mut(),
            };
            h = layer.forward_with_observer(
                layer_input,
                stream,
                &format!("model.layers.{i}"),
                observer,
            )?;
        }

        let output = self.norm.forward(&h, stream)?;
        observer.observe("model.norm", &output)?;
        Ok(output)
    }

    pub(crate) fn forward_expert_parallel<C>(
        &mut self,
        input: ModelInput<'_, C>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        mut observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        let mut hidden = self.embed_tokens.forward(inputs, stream)?;
        hidden = rms_norm_without_scale(&hidden, self.norm.eps, stream)?;
        let full_mask = full_attention_mask(&hidden, cache, mask.is_some(), stream)?;
        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }
        for (index, (layer, cache)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_mask = match mask {
                Some(mask) => Some(mask),
                None if layer.self_attn.sliding_window.is_none() => full_mask.as_ref(),
                None => None,
            };
            let layer_observer = observer
                .as_mut()
                .map(|observer| &mut **observer as &mut dyn ActivationObserver);
            hidden = layer.forward_expert_parallel(
                AttentionInput {
                    x: &hidden,
                    mask: layer_mask,
                    cache: cache.as_mut(),
                },
                assignment,
                group,
                statistics,
                &format!("model.layers.{index}"),
                layer_observer,
                stream,
            )?;
        }
        self.norm.forward(&hidden, stream)
    }
}

/// Input for a Muse-Glimmer forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for Decoder
where
    C: KeyValueCache + Default,
{
    type Output = Array;

    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs, stream)?;
        h = rms_norm_without_scale(&h, self.norm.eps, stream)?;

        let full_mask = full_attention_mask(&h, cache, mask.is_some(), stream)?;

        if cache.is_empty() {
            *cache = (0..self.layers.len()).map(|_| Some(C::default())).collect();
        }

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_mask = match mask {
                Some(mask) => Some(mask),
                None if layer.self_attn.sliding_window.is_none() => full_mask.as_ref(),
                None => None,
            };
            let layer_input = AttentionInput {
                x: &h,
                mask: layer_mask,
                cache: c.as_mut(),
            };
            h = layer.forward(layer_input, stream)?;
        }

        self.norm.forward(&h, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.embed_tokens.training_mode(mode);
        for layer in &mut self.layers {
            <TransformerBlock as Module<AttentionInput<'_, C>>>::training_mode(layer, mode);
        }
        self.norm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Qwen2/Qwen2.5/Qwen3 causal language model.
pub struct Model {
    /// Model configuration.
    pub args: DecoderConfig,

    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: Decoder,

    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl Model {
    /// Creates an unloaded Muse-Glimmer causal language model.
    pub fn new(args: DecoderConfig, stream: &Stream) -> Result<Self, Exception> {
        let model = Decoder::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.weight_quantization_for("lm_head.weight"),
                    stream,
                )?,
            )
        } else {
            None
        };

        Ok(Self {
            args,
            model,
            lm_head,
        })
    }

    /// Returns the configured model type.
    pub fn model_type(&self) -> &str {
        &self.args.model_type
    }

    /// Creates architecture-correct per-layer KV caches, including Qwen2 SWA layers.
    pub fn new_cache(&self) -> Vec<Option<crate::runtime::cache::ConcatKeyValueCache>> {
        self.args
            .attention_schedule
            .iter()
            .map(|policy| {
                Some(match policy.window() {
                    Some(window) => {
                        crate::runtime::cache::ConcatKeyValueCache::new_for_sliding_attention(
                            i32::try_from(window.get())
                                .expect("validated Muse-Glimmer attention window fits i32"),
                        )
                    }
                    None => crate::runtime::cache::ConcatKeyValueCache::new(),
                })
            })
            .collect()
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let out = self.model.forward_with_observer(input, stream, observer)?;
        observer.observe("model.output", &out)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &out,
            stream,
        )?;
        let logits = scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            stream,
        )?;
        observer.observe("lm_head.logits", &logits)?;
        Ok(logits)
    }

    pub(crate) fn forward_expert_parallel<C>(
        &mut self,
        input: ModelInput<'_, C>,
        assignment: &crate::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::architectures::distributed::expert::RoutingStatistics,
        observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
    {
        let hidden = self
            .model
            .forward_expert_parallel(input, assignment, group, statistics, observer, stream)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )?;
        scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            stream,
        )
    }

    /// Runs pure expert parallelism with externally supplied cache-backed experts.
    pub(crate) fn forward_cached_expert_parallel<C, F>(
        &mut self,
        input: ModelInput<'_, C>,
        mut execute: F,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache + Default,
        F: FnMut(usize, &Array, &Array, &Array, &Stream) -> Result<Array, Exception>,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;
        let mut hidden = self.model.embed_tokens.forward(inputs, stream)?;
        hidden = rms_norm_without_scale(&hidden, self.model.norm.eps, stream)?;
        let full_mask = full_attention_mask(&hidden, cache, mask.is_some(), stream)?;
        if cache.is_empty() {
            *cache = (0..self.model.layers.len())
                .map(|_| Some(C::default()))
                .collect();
        }
        for (index, (layer, layer_cache)) in self
            .model
            .layers
            .iter_mut()
            .zip(cache.iter_mut())
            .enumerate()
        {
            let layer_mask = match mask {
                Some(mask) => Some(mask),
                None if layer.self_attn.sliding_window.is_none() => full_mask.as_ref(),
                None => None,
            };
            hidden = layer.forward_sparse_experts(
                AttentionInput {
                    x: &hidden,
                    mask: layer_mask,
                    cache: layer_cache.as_mut(),
                },
                stream,
                |flat, indices, weights, stream| execute(index, flat, indices, weights, stream),
            )?;
        }
        let hidden = self.model.norm.forward(&hidden, stream)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &hidden,
            stream,
        )?;
        scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            stream,
        )
    }
}

impl<C> Module<ModelInput<'_, C>> for Model
where
    C: KeyValueCache + Default,
{
    type Output = Array;

    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        let out = self.model.forward(input, stream)?;
        let logits = project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &out,
            stream,
        )?;
        scale_logits(
            logits,
            self.args.output_multiplier,
            self.args.final_logit_softcapping,
            stream,
        )
    }

    fn training_mode(&mut self, mode: bool) {
        <Decoder as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

/// Loads `tokenizer.json` from a Muse-Glimmer model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads and validates a Muse-Glimmer decoder configuration from `config.json`.
pub fn load_config(model_dir: impl AsRef<Path>) -> Result<DecoderConfig, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let config: Value = serde_json::from_reader(file)?;
    config_from_hf_value(&config)
}

/// Parses and validates the arguments shared by structural preflight and loading.
pub(crate) fn config_from_hf_value(config: &Value) -> Result<DecoderConfig, Error> {
    let model_type = config
        .get("model_type")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture("Muse-Glimmer config is missing model_type".into())
        })?;
    if model_type != "muse_glimmer" {
        return Err(Error::UnsupportedModelType(model_type.into()));
    }
    validate_declared_architectures(config, model_type)?;
    let text_config = config.get("text_config").ok_or_else(|| {
        Error::UnsupportedArchitecture("Muse-Glimmer config is missing text_config".into())
    })?;
    validate_execution_fields(text_config)?;
    let mut source =
        serde_json::from_value::<DecoderConfigSource>(text_config.clone()).map_err(|error| {
            Error::UnsupportedArchitecture(format!("invalid Muse-Glimmer text_config: {error}"))
        })?;
    source.normalize_head_dim();
    let attention_schedule = muse_hf_attention_schedule(&source)?;
    let mut args = source.into_config(attention_schedule, WeightConvention::HuggingFace);
    args.vision_config = Some(vision::VisionConfig::from_hf_value(
        config.get("vision_config").ok_or_else(|| {
            Error::UnsupportedArchitecture("Muse-Glimmer config is missing vision_config".into())
        })?,
        args.hidden_size,
    )?);
    args.image_token_id = required_top_level_u32(config, "image_token_id")?;
    args.video_token_id = required_top_level_u32(config, "video_token_id")?;
    args.vision_out_hidden_size = required_top_level_i32(config, "out_hidden_size")?;
    args.projector_hidden_size = required_top_level_i32(config, "projector_hidden_size")?;
    if args.vision_out_hidden_size
        != args
            .vision_config
            .as_ref()
            .expect("assigned vision config")
            .hidden_size
            * 4
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer out_hidden_size {} must equal four times vision hidden_size",
            args.vision_out_hidden_size
        )));
    }
    validate_model_args(&args)?;
    Ok(args)
}

fn required_top_level_i32(config: &Value, name: &str) -> Result<i32, Error> {
    config
        .get(name)
        .and_then(Value::as_i64)
        .and_then(|value| i32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer config requires positive integer {name}"
            ))
        })
}

fn required_top_level_u32(config: &Value, name: &str) -> Result<u32, Error> {
    config
        .get(name)
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer config requires unsigned integer {name}"
            ))
        })
}

/// Validates the Muse-Glimmer portions of a Hugging Face model configuration.
pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    config_from_hf_value(config).map(|_| ())
}

fn muse_hf_attention_schedule(
    source: &DecoderConfigSource,
) -> Result<LayerSchedule<AttentionPolicy>, Error> {
    let layers = usize::try_from(source.num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "num_hidden_layers must be positive, got {}",
            source.num_hidden_layers
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "num_hidden_layers must be positive, got 0".into(),
        ));
    }
    if source.layer_types.len() != layers || source.layer_rope_theta.len() != layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer layer_types ({}) and layer_rope_theta ({}) must each match {layers} layers",
            source.layer_types.len(), source.layer_rope_theta.len()
        )));
    }
    let window = u32::try_from(source.sliding_window).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer sliding_window must be positive, got {}",
            source.sliding_window
        ))
    })?;
    if window == 0 || window > i32::MAX as u32 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer sliding_window is outside the executable range: {window}"
        )));
    }
    let sliding = AttentionPolicy::sliding(window)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let policies = source
        .layer_types
        .iter()
        .zip(&source.layer_rope_theta)
        .enumerate()
        .map(
            |(layer, (kind, theta))| match (kind.as_str(), *theta == 0.0) {
                ("sliding_attention", false) => Ok(sliding),
                ("full_attention", true) => Ok(AttentionPolicy::Full),
                _ => Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer layer {layer} has incompatible type {kind:?} and rope theta {theta}"
            ))),
            },
        )
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn required_positive_hf_u32(
    config: &Value,
    field: &'static str,
    context: &str,
) -> Result<u32, Error> {
    let value = config.get(field).ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("{context} requires a {field} value"))
    })?;
    let value = value.as_i64().ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("{field} must be a positive integer"))
    })?;
    if value <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "{field} must be positive, got {value}"
        )));
    }
    u32::try_from(value).map_err(|_| {
        Error::UnsupportedArchitecture(format!("{field} exceeds the u32 range: {value}"))
    })
}

fn required_nonnegative_hf_usize(
    config: &Value,
    field: &'static str,
    context: &str,
) -> Result<usize, Error> {
    let value = config
        .get(field)
        .ok_or_else(|| Error::UnsupportedArchitecture(format!("{context} requires {field}")))?;
    let value = value.as_i64().ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("{field} must be a non-negative integer"))
    })?;
    usize::try_from(value).map_err(|_| {
        Error::UnsupportedArchitecture(format!("{field} must be non-negative, got {value}"))
    })
}

fn validate_execution_fields(config: &Value) -> Result<(), Error> {
    for field in [
        "sliding_window_pattern",
        "attention_type",
        "use_qk_norm",
        "qk_norm",
        "attention_chunk_size",
        "value_head_dim",
        "attention_output_bias",
    ] {
        if config.get(field).is_some() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer config field {field:?} changes decoder execution and is not supported"
            )));
        }
    }
    if let Some(value) = config.get("partial_rotary_factor") {
        match value.as_f64() {
            Some(1.0) => {}
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(
                    "Muse-Glimmer requires full-head rotary embeddings (partial_rotary_factor=1)"
                        .into(),
                ));
            }
            None => {
                return Err(Error::UnsupportedArchitecture(
                    "partial_rotary_factor must be numeric".into(),
                ));
            }
        }
    }
    if let Some(value) = config.get("rope_interleaved") {
        match value.as_bool() {
            Some(false) => {}
            Some(true) => {
                return Err(Error::UnsupportedArchitecture(
                    "Muse-Glimmer safetensors do not declare interleaved RoPE coordinates".into(),
                ));
            }
            None => {
                return Err(Error::UnsupportedArchitecture(
                    "rope_interleaved must be boolean".into(),
                ));
            }
        }
    }
    Ok(())
}

fn validate_declared_architectures(config: &Value, model_type: &str) -> Result<(), Error> {
    let Some(value) = config.get("architectures") else {
        return Ok(());
    };
    let architectures = value.as_array().ok_or_else(|| {
        Error::UnsupportedArchitecture("config architectures must be an array of strings".into())
    })?;
    let expected: &[&str] = match model_type {
        "muse_glimmer" => &["MuseGlimmerForConditionalGeneration"],
        _ => return Err(Error::UnsupportedModelType(model_type.into())),
    };
    if architectures.is_empty()
        || architectures.iter().any(|architecture| {
            architecture
                .as_str()
                .is_none_or(|architecture| !expected.contains(&architecture))
        })
    {
        return Err(Error::UnsupportedArchitecture(format!(
            "model_type {model_type:?} requires architectures {expected:?}; multimodal, MoE, and custom-code variants are not interchangeable"
        )));
    }
    Ok(())
}

fn validate_model_args(args: &DecoderConfig) -> Result<(), Error> {
    if args.model_type != "muse_glimmer_text" {
        return Err(Error::UnsupportedModelType(args.model_type.clone()));
    }
    for (name, value) in [
        ("hidden_size", args.hidden_size),
        ("num_hidden_layers", args.num_hidden_layers),
        ("num_attention_heads", args.num_attention_heads),
        ("num_key_value_heads", args.num_key_value_heads),
        ("vocab_size", args.vocab_size),
        ("max_position_embeddings", args.max_position_embeddings),
        ("head_dim", args.head_dim),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "{name} must be positive, got {value}"
            )));
        }
    }
    if args.num_attention_heads % args.num_key_value_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            args.num_attention_heads, args.num_key_value_heads
        )));
    }
    args.num_attention_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "query projection width overflows i32: {} heads x head_dim {}",
                args.num_attention_heads, args.head_dim
            ))
        })?;
    args.num_key_value_heads
        .checked_mul(args.head_dim)
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "key/value projection width overflows i32: {} heads x head_dim {}",
                args.num_key_value_heads, args.head_dim
            ))
        })?;
    if !args.rms_norm_eps.is_finite() || args.rms_norm_eps <= 0.0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "rms_norm_eps must be finite and positive, got {}",
            args.rms_norm_eps
        )));
    }
    if !args.rope_theta.is_finite() || args.rope_theta <= 0.0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "rope_theta must be finite and positive, got {}",
            args.rope_theta
        )));
    }
    if args.hidden_act != "silu" {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer requires hidden_activation=\"silu\", got {:?}",
            args.hidden_act
        )));
    }
    if args.attention_dropout != 0.0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer inference requires attention_dropout=0, got {}",
            args.attention_dropout
        )));
    }
    if args.attention_bias == Some(true) {
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer text attention projections must be bias-free".into(),
        ));
    }
    if args.mlp_bias == Some(true) {
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer does not support biased SwiGLU projections".into(),
        ));
    }
    validate_attention_schedule(args)?;
    validate_rope_scaling(args)?;
    if args.intermediate_size <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "intermediate_size must be positive for Muse-Glimmer, got {}",
            args.intermediate_size
        )));
    }
    for (name, value) in [
        ("post_norm_eps", args.post_norm_eps),
        ("qk_scale_factor", args.qk_scale_factor),
        ("output_multiplier", args.output_multiplier),
        ("final_logit_softcapping", args.final_logit_softcapping),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "{name} must be finite and positive, got {value}"
            )));
        }
    }
    Ok(())
}

fn validate_attention_schedule(args: &DecoderConfig) -> Result<(), Error> {
    if args.attention_schedule.len() != args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "attention schedule has {} entries for {} decoder layers",
            args.attention_schedule.len(),
            args.num_hidden_layers
        )));
    }
    if args.layer_uses_rope.len() != args.attention_schedule.len() {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer RoPE schedule has {} entries for {} layers",
            args.layer_uses_rope.len(),
            args.attention_schedule.len()
        )));
    }
    for window in args
        .attention_schedule
        .iter()
        .copied()
        .filter_map(AttentionPolicy::window)
    {
        if window.get() > i32::MAX as u32 {
            return Err(Error::UnsupportedArchitecture(format!(
                "sliding attention window exceeds the executable i32 range: {}",
                window.get()
            )));
        }
    }
    Ok(())
}

fn validate_rope_scaling(args: &DecoderConfig) -> Result<(), Error> {
    let Some(config) = &args.rope_scaling else {
        return Ok(());
    };
    let rope_type = config
        .get("rope_type")
        .or_else(|| config.get("type"))
        .and_then(|value| match value {
            FloatOrString::String(value) => Some(value.as_str()),
            _ => None,
        })
        .ok_or_else(|| {
            Error::UnsupportedArchitecture(
                "rope_scaling must contain string field type or rope_type".into(),
            )
        })?;
    let allowed: &[&str] = match rope_type {
        "default" => &["type", "rope_type"],
        "linear" => &["type", "rope_type", "factor"],
        "yarn" => &[
            "type",
            "rope_type",
            "factor",
            "original_max_position_embeddings",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
            "truncate",
        ],
        other => {
            return Err(Error::UnsupportedArchitecture(format!(
                "dense Qwen RoPE scaling type {other:?} is unsupported"
            )));
        }
    };
    if let Some(key) = config.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(Error::UnsupportedArchitecture(format!(
            "dense Qwen RoPE scaling field {key:?} affects unsupported execution semantics"
        )));
    }
    let numeric = |key: &str| {
        config.get(key).and_then(|value| match value {
            FloatOrString::Float(value) if value.is_finite() => Some(*value),
            FloatOrString::String(value) => value.parse::<f32>().ok().filter(|v| v.is_finite()),
            _ => None,
        })
    };
    if matches!(rope_type, "linear" | "yarn")
        && numeric("factor").is_none_or(|factor| factor <= 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "scaled Muse-Glimmer RoPE requires a finite positive factor".into(),
        ));
    }
    if rope_type == "yarn"
        && numeric("original_max_position_embeddings").is_none_or(|value| value <= 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "YaRN Muse-Glimmer RoPE requires positive original_max_position_embeddings".into(),
        ));
    }
    Ok(())
}

pub(crate) struct LoadedGguf {
    pub(crate) model: Model,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Official image-only Muse-Glimmer GGUF vision sidecar.
pub(crate) struct MuseGlimmerMmprojGguf {
    pub(crate) checkpoint: GgufCheckpoint,
    pub(crate) metadata: HashMap<String, GgufMetadataValue>,
    pub(crate) path: PathBuf,
}

pub(crate) fn open_sibling_mmproj(
    gguf_file: &Path,
) -> Result<Option<MuseGlimmerMmprojGguf>, Error> {
    let Some(path) =
        crate::runtime::checkpoint::gguf::find_sibling_mmproj(gguf_file, "muse-glimmer")?
    else {
        return Ok(None);
    };
    let checkpoint = GgufCheckpoint::open(&path)?;
    let metadata = gguf_metadata(&checkpoint);
    vision::VisionConfig::from_gguf_metadata(&metadata, 6656)?;
    checkpoint
        .catalog()
        .translated_outputs(translate_mmproj_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    Ok(Some(MuseGlimmerMmprojGguf {
        checkpoint,
        metadata,
        path,
    }))
}

pub(crate) fn translate_mmproj_weight_name(name: &str) -> String {
    let exact = match name {
        "v.patch_embd.weight" => Some("vision_tower.patch_embedder.patch_embedding.weight"),
        "v.position_embd.weight" => {
            Some("vision_tower.patch_embedder.position_embedding_table.weight")
        }
        "v.pre_ln.weight" => Some("vision_tower.ln_pre.weight"),
        "v.pre_ln.bias" => Some("vision_tower.ln_pre.bias"),
        "v.post_ln.weight" => Some("vision_tower.ln_post.weight"),
        "v.post_ln.bias" => Some("vision_tower.ln_post.bias"),
        "mm.0.weight" => Some("vision_adapter.fc1.weight"),
        "mm.1.weight" => Some("vision_adapter.fc2.weight"),
        "mm.2.weight" => Some("vision_projection.weight"),
        _ => None,
    };
    if let Some(target) = exact {
        return target.into();
    }
    let Some(rest) = name.strip_prefix("v.blk.") else {
        return name.into();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.into();
    };
    for (source, target) in [
        ("attn_q", "attn.q_proj"),
        ("attn_k", "attn.k_proj"),
        ("attn_v", "attn.v_proj"),
        ("attn_out", "attn.proj"),
        ("ffn_up", "mlp.fc1"),
        ("ffn_down", "mlp.fc2"),
        ("ln1", "norm1"),
        ("ln2", "norm2"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "vision_tower.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.into()
}

pub(crate) fn translate_mmproj_store_weight_name(name: &str) -> String {
    format!("model.{}", translate_mmproj_weight_name(name))
}

/// Loads a Muse-Glimmer GGUF checkpoint.
///
/// Dense tensors and GGUF Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0 tensors are
/// supported. The quantized formats are consumed in the packed affine
/// representation emitted by MLX's GGUF loader.
pub fn load_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    Ok(load_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

pub(crate) fn load_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGguf, Error> {
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    load_gguf_checkpoint(&checkpoint, metadata, None, stream, weights_stream)
}

pub(crate) fn load_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedGguf, Error> {
    let architecture = gguf_string(&metadata, "general.architecture")?;
    if architecture != "muse-glimmer" {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports muse-glimmer"
        )));
    }
    let is_moe = false;
    let gguf_architecture = crate::api::GgufArchitecture::resolve(&architecture)?;
    crate::api::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        &metadata,
        crate::api::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let translate = |name: &str| translate_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut args = config_from_gguf_catalog(checkpoint, &metadata, &architecture, is_moe)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate)?;
    if is_moe {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts");
            if let Some(config) = configs.remove(&format!("{prefix}.gate_proj")) {
                configs.remove(&format!("{prefix}.up_proj"));
                configs.insert(format!("{prefix}.gate_up_proj"), config);
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    if let Some(quantization) = quantization {
        args.quantization = Some(quantization);
        args.quantization_config = None;
        args.quantized_weights = None;
        args.quantized_weight_configs = None;
    }

    let mut model = Model::new(args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    if !is_moe {
        load_gguf_strict(
            &mut model,
            checkpoint,
            quantization.map(|value| (value, stream)),
            &config,
            &mut report,
            |name, value| Ok((translate_gguf_weight_name(&name, false), value)),
        )?;
    } else {
        let mut materializer = checkpoint.materializer();
        for tensor in checkpoint.catalog().tensors() {
            let physical_name = &tensor.descriptor().name;
            if physical_name.contains("ffn_gate_exps") || physical_name.contains("ffn_up_exps") {
                continue;
            }
            for (name, value) in materializer.converted_tensor(physical_name)?.into_arrays() {
                load_named_array_strict(
                    &mut model,
                    translate_gguf_weight_name(&name, true),
                    value,
                    quantization.map(|value| (value, stream)),
                    &config,
                    &mut report,
                )?;
            }
        }
        for layer in 0..model.args.num_hidden_layers {
            let source_prefix = format!("blk.{layer}");
            let target_prefix = format!("model.layers.{layer}.mlp.experts");
            let gate = materializer
                .converted_tensor(&format!("{source_prefix}.ffn_gate_exps.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            let up = materializer
                .converted_tensor(&format!("{source_prefix}.ffn_up_exps.weight"))?
                .into_arrays()
                .into_iter()
                .collect::<HashMap<_, _>>();
            for (source_suffix, target_suffix) in
                [("weight", ""), ("scales", "_scales"), ("biases", "_biases")]
            {
                let gate_name = format!("{source_prefix}.ffn_gate_exps.{source_suffix}");
                let up_name = format!("{source_prefix}.ffn_up_exps.{source_suffix}");
                match (gate.get(&gate_name), up.get(&up_name)) {
                    (Some(gate), Some(up)) => {
                        let value = concatenate_axis(&[gate.clone(), up.clone()], 1, weights_stream)?;
                        load_named_array_strict(
                            &mut model,
                            format!("{target_prefix}.gate_up_proj{target_suffix}"),
                            value,
                            quantization.map(|value| (value, stream)),
                            &config,
                            &mut report,
                        )?;
                    }
                    (None, None) if source_suffix != "weight" => {}
                    _ => {
                        return Err(Error::UnsupportedArchitecture(format!(
                            "Qwen3 MoE GGUF has incomplete gate/up expert tensors under {source_prefix}"
                        )))
                    }
                }
            }
        }
    }
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    let eos_token_ids = crate::api::gguf_eos_token_ids(&metadata)?;
    Ok(LoadedGguf {
        model,
        eos_token_ids,
    })
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
) -> Result<(DecoderConfig, Vec<u32>), Error> {
    let gguf_architecture = crate::api::GgufArchitecture::resolve(architecture)?;
    crate::api::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default(),
    )
    .into_loader_result()?;
    let translate = |name: &str| translate_gguf_weight_name(name, is_moe);
    checkpoint
        .catalog()
        .translated_outputs(translate)
        .map_err(safemlx::error::IoError::from)?;
    let mut args = config_from_gguf_catalog(checkpoint, metadata, architecture, is_moe)?;
    let mut configs = gguf_quantization_configs(checkpoint, translate)?;
    if is_moe {
        for layer in 0..args.num_hidden_layers {
            let prefix = format!("model.layers.{layer}.mlp.experts");
            if let Some(config) = configs.remove(&format!("{prefix}.gate_proj")) {
                configs.remove(&format!("{prefix}.up_proj"));
                configs.insert(format!("{prefix}.gate_up_proj"), config);
            }
        }
    }
    args.quantized_weights = Some(configs.keys().cloned().collect());
    args.quantized_weight_configs = Some(configs);
    args.quantization = None;
    let eos_token_ids = crate::api::gguf_eos_token_ids(metadata)?;
    Ok((args, eos_token_ids))
}

/// Parses the GGUF arguments shared by structural preflight and loading.
pub(crate) fn config_from_gguf_catalog(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
) -> Result<DecoderConfig, Error> {
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let hidden_size = gguf_i32_catalog(metadata, &key("embedding_length"))?;
    let num_hidden_layers = gguf_i32_catalog(metadata, &key("block_count"))?;
    let num_attention_heads = gguf_i32_catalog(metadata, &key("attention.head_count"))?;
    if num_attention_heads <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF attention head count must be positive, got {num_attention_heads}"
        )));
    }
    let num_key_value_heads = gguf_optional_i64(metadata, &key("attention.head_count_kv"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| Error::UnsupportedArchitecture("GGUF KV-head count exceeds i32".into()))?
        .unwrap_or(num_attention_heads);
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| {
            Error::UnsupportedArchitecture("GGUF attention key length exceeds i32".into())
        })?
        .unwrap_or(hidden_size / num_attention_heads);
    if let Some(value_head_dim) = gguf_optional_i64(metadata, &key("attention.value_length"))? {
        if value_head_dim != i64::from(head_dim) {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF attention value length {value_head_dim} does not match key/head length {head_dim}"
            )));
        }
    }
    if let Some(rotary_dim) = gguf_optional_i64(metadata, &key("rope.dimension_count"))? {
        if rotary_dim != i64::from(head_dim) {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF rotary dimension count {rotary_dim} does not match the full head dimension {head_dim}"
            )));
        }
    }
    if matches!(
        gguf_optional_bool(metadata, &key("attention.causal"))?,
        Some(false)
    ) {
        return Err(Error::UnsupportedArchitecture(
            "dense Qwen GGUF checkpoints must use causal attention".into(),
        ));
    }
    if let Some(scale) = gguf_optional_f32(metadata, &key("attention.scale"))? {
        let expected = 1.0 / (head_dim as f32).sqrt();
        if !scale.is_finite() || (scale - expected).abs() > expected.abs().max(1.0) * 1e-6 {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF attention scale {scale} is unsupported; dense Qwen requires {expected} for head dimension {head_dim}"
            )));
        }
    }
    if let Some(activation) = gguf_optional_string(metadata, &key("hidden_activation"))? {
        if activation != "silu" {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF hidden activation {activation:?} is unsupported; dense Qwen requires \"silu\""
            )));
        }
    }
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(GgufMetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len()).map_err(|_| {
            Error::UnsupportedArchitecture("GGUF tokenizer vocabulary exceeds i32".into())
        })?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(Error::UnsupportedArchitecture(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type".into(),
            ));
        }
        None => gguf_i32_catalog(metadata, &key("vocab_size"))?,
    };

    if architecture != "muse-glimmer" || is_moe {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer GGUF config parser cannot load architecture {architecture:?} as MoE={is_moe}"
        )));
    }
    let attention_schedule =
        muse_gguf_attention_schedule(metadata, architecture, num_hidden_layers)?;
    let layer_uses_rope = attention_schedule
        .iter()
        .map(|policy| policy.window().is_some())
        .collect::<Vec<_>>();
    let args = DecoderConfig {
        model_type: "muse_glimmer".to_string(),
        hidden_size,
        num_hidden_layers,
        intermediate_size: if is_moe {
            gguf_optional_i64(metadata, &key("feed_forward_length"))?
                .map(i32::try_from)
                .transpose()
                .map_err(|_| {
                    Error::UnsupportedArchitecture("GGUF feed-forward length exceeds i32".into())
                })?
                .unwrap_or(0)
        } else {
            gguf_i32_catalog(metadata, &key("feed_forward_length"))?
        },
        num_attention_heads,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        post_norm_eps: gguf_optional_f32(metadata, &key("attention.post_norm_rms_epsilon"))?
            .unwrap_or(1e-8),
        vocab_size,
        num_key_value_heads,
        max_position_embeddings: gguf_i32_catalog(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(1_000_000.0),
        layer_uses_rope,
        head_dim,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        rope_scaling: gguf_rope_scaling(metadata, architecture)?,
        hidden_act: default_hidden_act(),
        attention_dropout: 0.0,
        attention_bias: Some(false),
        mlp_bias: Some(false),
        attention_schedule,
        qk_scale_factor: 3.87,
        output_multiplier: gguf_f32(metadata, &key("logit_scale"))?,
        final_logit_softcapping: gguf_f32(metadata, &key("final_logit_softcapping"))?,
        weight_convention: WeightConvention::Gguf,
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
        moe_intermediate_size: if is_moe {
            gguf_i32_catalog(metadata, &key("expert_feed_forward_length"))?
        } else {
            0
        },
        num_experts: if is_moe {
            gguf_i32_catalog(metadata, &key("expert_count"))?
        } else {
            0
        },
        num_experts_per_tok: if is_moe {
            gguf_i32_catalog(metadata, &key("expert_used_count"))?
        } else {
            0
        },
        norm_topk_prob: is_moe,
        quantized_weight_configs: None,
        vision_config: None,
        image_token_id: 200092,
        video_token_id: 200091,
        vision_out_hidden_size: 6144,
        projector_hidden_size: 4096,
    };
    validate_model_args(&args)?;
    Ok(args)
}

fn muse_gguf_attention_schedule(
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    layers: i32,
) -> Result<LayerSchedule<AttentionPolicy>, Error> {
    let layers = usize::try_from(layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer GGUF decoder layer count must be positive, got {layers}"
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer GGUF decoder layer count must be positive, got 0".into(),
        ));
    }
    let window_key = format!("{architecture}.attention.sliding_window");
    let window = gguf_optional_i64(metadata, &window_key)?;
    let pattern_key = format!("{architecture}.attention.sliding_window_pattern");
    let pattern = match metadata.get(&pattern_key) {
        None => None,
        Some(GgufMetadataValue::Array(safemlx::ops::GgufMetadataArray::Bool(values))) => {
            Some(values.as_slice())
        }
        Some(_) => {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF metadata key {pattern_key:?} must be a boolean array"
            )));
        }
    };
    if let Some(pattern) = pattern {
        if pattern.len() != layers {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer GGUF sliding-window pattern has {} entries for {layers} layers",
                pattern.len()
            )));
        }
    }
    let window = match window {
        Some(window) if window <= 0 => {
            return Err(Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer GGUF sliding window must be positive, got {window}"
            )))
        }
        Some(window) => Some(u32::try_from(window).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "Muse-Glimmer GGUF sliding window exceeds the u32 range: {window}"
            ))
        })?),
        None => None,
    };
    if window.is_some_and(|window| window > i32::MAX as u32) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Muse-Glimmer GGUF sliding window exceeds the executable i32 range: {}",
            window.expect("checked present window")
        )));
    }
    let Some(window) = window else {
        if pattern.is_some_and(|pattern| pattern.iter().any(|value| *value)) {
            return Err(Error::UnsupportedArchitecture(format!(
                "GGUF metadata {pattern_key:?} enables sliding layers without {window_key:?}"
            )));
        }
        return LayerSchedule::all_full(layers)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
    };
    let pattern = match pattern {
        None => vec![true; layers],
        Some(pattern) => pattern.to_vec(),
    };
    LayerSchedule::from_sliding_pattern(layers, &pattern, Some(window))
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))
}

fn gguf_rope_scaling(
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
) -> Result<Option<HashMap<String, FloatOrString>>, Error> {
    let scaling_type_key = format!("{architecture}.rope.scaling.type");
    let Some(scaling_type) = gguf_optional_string(metadata, &scaling_type_key)? else {
        return Ok(None);
    };
    match scaling_type.as_str() {
        "none" | "default" => Ok(None),
        "linear" => {
            let factor_key = format!("{architecture}.rope.scaling.factor");
            let factor = gguf_optional_f32(metadata, &factor_key)?.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "linear GGUF RoPE scaling is missing {factor_key}"
                ))
            })?;
            Ok(Some(HashMap::from([
                (
                    "rope_type".to_string(),
                    FloatOrString::String("linear".to_string()),
                ),
                ("factor".to_string(), FloatOrString::Float(factor)),
            ])))
        }
        "yarn" => {
            // GGUF uses architecture-scoped names which differ from the
            // Hugging Face map consumed by `initialize_rope`.
            for suffix in ["yarn_ext_factor", "yarn_attn_factor", "yarn_log_multiplier"] {
                let key = format!("{architecture}.rope.scaling.{suffix}");
                if metadata.contains_key(&key) {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "GGUF YaRN field {key:?} changes attention scaling semantics that the Muse-Glimmer decoder does not implement"
                    )));
                }
            }
            let factor = gguf_f32(metadata, &format!("{architecture}.rope.scaling.factor"))?;
            let original = gguf_i32_catalog(
                metadata,
                &format!("{architecture}.rope.scaling.original_context_length"),
            )? as f32;
            let beta_fast = gguf_optional_f32(
                metadata,
                &format!("{architecture}.rope.scaling.yarn_beta_fast"),
            )?
            .unwrap_or(32.0);
            let beta_slow = gguf_optional_f32(
                metadata,
                &format!("{architecture}.rope.scaling.yarn_beta_slow"),
            )?
            .unwrap_or(1.0);
            Ok(Some(HashMap::from([
                (
                    "rope_type".to_string(),
                    FloatOrString::String("yarn".to_string()),
                ),
                ("factor".to_string(), FloatOrString::Float(factor)),
                (
                    "original_max_position_embeddings".to_string(),
                    FloatOrString::Float(original),
                ),
                ("beta_fast".to_string(), FloatOrString::Float(beta_fast)),
                ("beta_slow".to_string(), FloatOrString::Float(beta_slow)),
                ("truncate".to_string(), FloatOrString::Bool(false)),
            ])))
        }
        other => Err(Error::UnsupportedArchitecture(format!(
            "GGUF RoPE scaling type {other:?} is not supported by the Muse-Glimmer GGUF loader"
        ))),
    }
}

pub(crate) fn translate_gguf_weight_name(name: &str, is_moe: bool) -> String {
    const ROOTS: [(&str, &str); 3] = [
        ("token_embd", "model.embed_tokens"),
        ("output_norm", "model.norm"),
        ("output", "lm_head"),
    ];
    for (source, target) in ROOTS {
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
    if is_moe {
        return name.to_string();
    }

    const PARAMETERS: [(&str, &str); 15] = [
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_gate", "self_attn.gate_proj"),
        ("attn_norm", "input_layernorm"),
        ("post_attention_norm", "post_attention_layernorm"),
        ("ffn_norm", "pre_feedforward_layernorm"),
        ("post_ffw_norm", "post_feedforward_layernorm"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_down", "mlp.down_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("rope_freqs", "rope_freqs"),
    ];
    for (source, target) in PARAMETERS {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!(
                "model.layers.{layer}.{}",
                parameter.replacen(source, target, 1)
            );
        }
    }
    name.to_string()
}

pub(crate) fn gguf_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<String, Error> {
    gguf_optional_string(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_string(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<String>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
    }
}

pub(crate) fn gguf_i32_catalog(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<i32, Error> {
    let value = gguf_i64(metadata, key)?;
    i32::try_from(value).map_err(|_| {
        Error::UnsupportedArchitecture(format!("GGUF metadata value {key:?} exceeds i32"))
    })
}

fn gguf_i64(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<i64, Error> {
    gguf_optional_i64(metadata, key)?.ok_or_else(|| {
        Error::UnsupportedArchitecture(format!("GGUF metadata is missing required key {key:?}"))
    })
}

fn gguf_optional_i64(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<i64>, Error> {
    match metadata.get(key) {
        Some(value) => value.as_i64().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
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
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!("GGUF metadata key {key:?} has the wrong type"))
        }),
        None => Ok(None),
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn tiny_config() -> Value {
        json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "image_token_id": 30,
            "video_token_id": 29,
            "out_hidden_size": 16,
            "projector_hidden_size": 8,
            "vision_config": {
                "model_type": "muse_glimmer_vision",
                "hidden_act": "gelu",
                "hidden_size": 4,
                "intermediate_size": 8,
                "num_attention_heads": 1,
                "num_hidden_layers": 1,
                "patch_size": 2,
                "patch_temporal": 2,
                "merge_size": 2,
                "pos_emb_height": 2,
                "pos_emb_width": 2,
                "max_position_embeddings": 4,
                "layer_norm_eps": 1e-5,
                "layer_types": ["full_attention"],
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "text_config": {
                "model_type": "muse_glimmer_text",
                "hidden_size": 24,
                "num_hidden_layers": 4,
                "intermediate_size": 48,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 4,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5,
                "post_norm_eps": 1e-8,
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
                "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "full_attention"],
                "layer_rope_theta": [500000.0, 500000.0, 500000.0, 0.0],
                "sliding_window": 16,
                "tie_word_embeddings": false,
                "hidden_activation": "silu",
                "attention_dropout": 0.0,
                "attention_bias": false,
                "mlp_bias": false,
                "qk_scale_factor": 3.87,
                "output_multiplier": 0.19611613513818404,
                "final_logit_softcapping": 20.0
            }
        })
    }

    #[test]
    fn parses_release_declared_mixed_attention_and_narrow_queries() {
        let args = config_from_hf_value(&tiny_config()).unwrap();
        assert_eq!(args.hidden_size, 24);
        assert_eq!(args.num_attention_heads * args.head_dim, 16);
        assert_eq!(args.layer_uses_rope, [true, true, true, false]);
        assert_eq!(args.attention_schedule.full_layer_count(), 1);
        assert_eq!(
            args.attention_schedule
                .sliding_windows()
                .get(&std::num::NonZeroU32::new(16).unwrap()),
            Some(&3)
        );
    }

    #[test]
    fn rejects_inconsistent_nope_declaration() {
        let mut config = tiny_config();
        config["text_config"]["layer_rope_theta"][3] = json!(500000.0);
        let error = config_from_hf_value(&config).unwrap_err();
        assert!(error.to_string().contains("incompatible type"));
    }

    #[test]
    fn translates_every_official_text_gguf_tensor_role() {
        let cases = [
            (
                "blk.7.attn_gate.weight",
                "model.layers.7.self_attn.gate_proj.weight",
            ),
            (
                "blk.7.ffn_norm.weight",
                "model.layers.7.pre_feedforward_layernorm.weight",
            ),
            (
                "blk.7.post_attention_norm.weight",
                "model.layers.7.post_attention_layernorm.weight",
            ),
            (
                "blk.7.post_ffw_norm.weight",
                "model.layers.7.post_feedforward_layernorm.weight",
            ),
            (
                "blk.7.attn_q_norm.weight",
                "model.layers.7.self_attn.q_norm.weight",
            ),
        ];
        for (source, expected) in cases {
            assert_eq!(translate_gguf_weight_name(source, false), expected);
        }
    }

    #[test]
    fn translates_official_projector_tensor_roles_without_unpacking() {
        let cases = [
            (
                "v.patch_embd.weight",
                "vision_tower.patch_embedder.patch_embedding.weight",
            ),
            (
                "v.blk.49.attn_q.weight",
                "vision_tower.layers.49.attn.q_proj.weight",
            ),
            (
                "v.blk.7.ffn_down.weight",
                "vision_tower.layers.7.mlp.fc2.weight",
            ),
            ("mm.0.weight", "vision_adapter.fc1.weight"),
            ("mm.1.weight", "vision_adapter.fc2.weight"),
            ("mm.2.weight", "vision_projection.weight"),
        ];
        for (source, expected) in cases {
            assert_eq!(translate_mmproj_weight_name(source), expected);
        }
        assert_eq!(
            translate_mmproj_store_weight_name("mm.2.weight"),
            "model.vision_projection.weight"
        );
    }

    #[test]
    fn strictly_binds_release_safetensors_text_namespace() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let args = config_from_hf_value(&tiny_config()).unwrap();
        let source = Model::new(args.clone(), stream).unwrap();
        let mut checkpoint = source
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| {
                let name = name.strip_prefix("model.").map_or_else(
                    || name.to_string(),
                    |rest| format!("model.language_model.{rest}"),
                );
                (name, value.clone())
            })
            .collect::<HashMap<_, _>>();
        checkpoint.insert(
            "model.vision_projection.weight".into(),
            Array::from_slice(&[0.0_f32], &[1]),
        );

        let mut loaded = Model::new(args, stream).unwrap();
        let config = safetensors_strict_load_config();
        let mut report = StrictLoadReport::default();
        crate::runtime::checkpoint::load::load_arrays_strict(
            &mut loaded,
            checkpoint,
            &config,
            &mut report,
        )
        .unwrap();
        report.finish(&loaded, &config).unwrap();
    }

    #[test]
    fn ordinary_block_matches_observed_centered_norm_execution() {
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let args = config_from_hf_value(&tiny_config()).unwrap();
        let mut direct = TransformerBlock::new_for_layer(&args, 0, stream).unwrap();
        for parameter in direct.parameters_mut().flatten().values_mut() {
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter = safemlx::ops::ones_dtype(&shape, dtype, stream).unwrap();
        }
        let mut observed = direct.clone();
        let values = (0..args.hidden_size)
            .map(|index| (index as f32 - 11.0) / 7.0)
            .collect::<Vec<_>>();
        let input = Array::from_slice(&values, &[1, 1, args.hidden_size]);

        let actual = direct
            .forward(
                AttentionInput::<ConcatKeyValueCache> {
                    x: &input,
                    mask: None,
                    cache: None,
                },
                stream,
            )
            .unwrap();
        let expected = observed
            .forward_with_observer::<ConcatKeyValueCache>(
                AttentionInput {
                    x: &input,
                    mask: None,
                    cache: None,
                },
                stream,
                "model.language_model.layers.0",
                &mut crate::runtime::execution::inspection::NoopObserver,
            )
            .unwrap();
        safemlx::transforms::eval([&actual, &expected]).unwrap();
        for (actual, expected) in actual
            .evaluated()
            .unwrap()
            .as_slice::<f32>()
            .iter()
            .zip(expected.evaluated().unwrap().as_slice::<f32>())
        {
            assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
        }
    }

    #[test]
    fn bfloat16_norms_preserve_activation_dtype() {
        use half::bf16;

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let input = Array::from_slice(
            &[
                bf16::from_f32(1.0),
                bf16::from_f32(-2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(-4.0),
            ],
            &[1, 1, 4],
        );

        let weightless = rms_norm_without_scale(&input, 1e-6, stream).unwrap();
        assert_eq!(weightless.dtype(), Dtype::Bfloat16);

        let mut centered = nn::RmsNorm::new(4).unwrap();
        centered.eps = 1e-6;
        *centered.weight.as_mut() = Array::from_slice(&[bf16::from_f32(0.0); 4], &[4]);
        let centered = forward_layer_norm(&mut centered, true, &input, stream).unwrap();
        assert_eq!(centered.dtype(), Dtype::Bfloat16);
    }

    #[test]
    fn bfloat16_block_preserves_activation_dtype() {
        use half::bf16;

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let args = config_from_hf_value(&tiny_config()).unwrap();
        let mut block = TransformerBlock::new_for_layer(&args, 0, stream).unwrap();
        for parameter in block.parameters_mut().flatten().values_mut() {
            let shape = parameter.shape().to_vec();
            **parameter = Array::from_slice(&vec![bf16::from_f32(1.0); parameter.size()], &shape);
        }
        let input = Array::from_slice(
            &(0..args.hidden_size)
                .map(|index| bf16::from_f32((index as f32 - 11.0) / 7.0))
                .collect::<Vec<_>>(),
            &[1, 1, args.hidden_size],
        );
        let output = block
            .forward(
                AttentionInput::<ConcatKeyValueCache> {
                    x: &input,
                    mask: None,
                    cache: None,
                },
                stream,
            )
            .unwrap();
        assert_eq!(output.dtype(), Dtype::Bfloat16);
    }
}

fn gguf_optional_bool(
    metadata: &HashMap<String, GgufMetadataValue>,
    key: &str,
) -> Result<Option<bool>, Error> {
    match metadata.get(key) {
        Some(GgufMetadataValue::Bool(value)) => Ok(Some(*value)),
        Some(_) => Err(Error::UnsupportedArchitecture(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Ok(None),
    }
}

#[derive(Debug, Clone, Deserialize)]
/// Hugging Face safetensors index file.
pub struct WeightMap {
    /// Index metadata.
    pub metadata: HashMap<String, Value>,
    /// Mapping from tensor name to shard file name.
    pub weight_map: HashMap<String, String>,
}

/// Loads a Muse-Glimmer model and SafeTensors weights from a model directory.
pub fn load_safetensors(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let model_args = load_config(model_dir)?;
    crate::api::structural::validate_safetensors_load_path(
        model_args.model_kind(),
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let mut model = Model::new(model_args, stream)?;
    let config = safetensors_strict_load_config();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_strict(&mut model, model_dir, weights_stream, &config, &mut report)?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;

    Ok(model)
}

/// Loads a Muse-Glimmer checkpoint while quantizing matrices tensor-by-tensor.
pub fn load_safetensors_quantized(
    model_dir: impl AsRef<Path>,
    quantization: WeightQuantization,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<Model, Error> {
    let model_dir = model_dir.as_ref();
    let mut model_args = load_config(model_dir)?;
    crate::api::structural::validate_safetensors_load_path(
        model_args.model_kind(),
        model_dir,
        crate::api::ModelLoadOptions::with_quantization(quantization),
    )?;
    if !crate::runtime::checkpoint::quantization::should_quantize_on_load(
        "Muse-Glimmer",
        model_args.weight_quantization(),
        quantization,
    )? {
        return load_safetensors(model_dir, stream, weights_stream);
    }
    model_args.quantization = Some(quantization);
    let mut model = Model::new(model_args, stream)?;
    let config = safetensors_strict_load_config();
    let mut report = StrictLoadReport::default();
    load_safetensors_dir_quantized_strict(
        &mut model,
        model_dir,
        weights_stream,
        stream,
        quantization,
        &config,
        &mut report,
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

fn safetensors_strict_load_config() -> StrictLoadConfig {
    StrictLoadConfig::default()
        .rewrite_prefix("model.language_model.", "model.")
        // The resident causal-LM object owns only the text model. The processor's
        // architecture adapter loads these independently for multimodal requests.
        .allow_unused_prefix("model.vision_tower.")
        .allow_unused_prefix("model.vision_adapter.")
        .allow_unused_prefix("model.vision_projection.")
}

impl<C> CausalLm<Vec<Option<C>>> for Model
where
    C: KeyValueCache + Default,
{
    fn prefill_input_logits(
        &mut self,
        input: input::ModelInput<'_>,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let prompt_tokens = input::text_token_ids(input, stream)?;
        let logits = self.forward(
            ModelInput {
                inputs: &prompt_tokens,
                mask: None,
                cache,
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }

    fn decode_logits(
        &mut self,
        input_tokens: &Array,
        cache: &mut Vec<Option<C>>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let logits = self.forward(
            ModelInput {
                inputs: input_tokens,
                mask: None,
                cache,
            },
            stream,
        )?;
        logits.try_index_device((.., -1, ..), stream)
    }
}

/// Dense-Qwen token generation iterator.
pub type Generate<'a, C, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Vec<Option<C>>, S>;
