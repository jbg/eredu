//! Shared Qwen2/Qwen2.5/Qwen3 decoder-only model implementation.

/// Architecture-owned physical checkpoint contracts.
#[path = "dense/checkpoint.rs"]
pub(crate) mod checkpoint;
/// Bounded and unified residency execution for dense Qwen decoders.
#[path = "dense/layerwise.rs"]
pub mod layerwise;

use eredu_checkpoint::WeightQuantization;
use eredu_nn::RopeValue;

use std::{
    collections::{BTreeMap, HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParameters as ModuleParametersTrait},
    nn,
    ops::{indexing::TryIndexOp, GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use crate::core::cache::{
    PromptCacheDescriptor, PromptCacheManifest, PromptCacheModelIdentity, PromptCacheOptions,
};

pub use crate::backend::mlx::nn::generation::sample;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::tensor::{
        create_causal_mask,
        rope::{initialize_rope, RopeVariant},
    },
    backend::mlx::nn::{
        self as common,
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
    backend::mlx::runtime::cache::{
        residency::{
            open_prompt_cache_snapshot, save_prompt_cache_snapshot, CacheBlockArrays,
            CacheResidencyManager, PromptCacheSnapshotBlock,
        },
        ConcatKeyValueCache, KeyValueCache, PagedKeyValueCache,
    },
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    backend::mlx::runtime::execution::inspection::{ActivationObserver, MoeRoutingObservation},
    backend::mlx::runtime::media::input,
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::CacheRankIdentity,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
/// Dense Qwen decoder generation selected by checkpoint metadata.
pub enum Architecture {
    /// Qwen2 and Qwen2.5 text decoders with biased Q/K/V projections.
    Qwen2,
    /// Qwen3 dense or sparse-MoE text decoders with Q/K normalization.
    Qwen3,
}

/// Builds the authoritative per-layer paged KV layout for any dense-Qwen
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
                i32::try_from(window.get()).expect("validated dense-Qwen attention window fits i32")
            });
            PagedKeyValueCache::new_with_layout(manager.clone(), layer, window, 0, rank).map(Some)
        })
        .collect()
}

#[derive(Debug, Clone)]
/// Validated dense-Qwen decoder geometry normalized from checkpoint metadata.
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
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// RoPE base frequency.
    pub rope_theta: f32,
    /// Per-head attention dimension. Zero is normalized from hidden/head geometry.
    pub head_dim: i32,
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    /// Optional RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
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
    vocab_size: i32,
    num_key_value_heads: i32,
    max_position_embeddings: i32,
    rope_theta: f32,
    #[serde(default)]
    head_dim: i32,
    tie_word_embeddings: bool,
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default = "default_hidden_act")]
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

    fn into_config(self, attention_schedule: LayerSchedule<AttentionPolicy>) -> DecoderConfig {
        DecoderConfig {
            model_type: self.model_type,
            hidden_size: self.hidden_size,
            num_hidden_layers: self.num_hidden_layers,
            intermediate_size: self.intermediate_size,
            num_attention_heads: self.num_attention_heads,
            rms_norm_eps: self.rms_norm_eps,
            vocab_size: self.vocab_size,
            num_key_value_heads: self.num_key_value_heads,
            max_position_embeddings: self.max_position_embeddings,
            rope_theta: self.rope_theta,
            head_dim: self.head_dim,
            tie_word_embeddings: self.tie_word_embeddings,
            rope_scaling: self.rope_scaling,
            hidden_act: self.hidden_act,
            attention_dropout: self.attention_dropout,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            attention_schedule,
            quantization: self.quantization,
            quantization_config: self.quantization_config,
            quantized_weights: None,
            moe_intermediate_size: self.moe_intermediate_size,
            num_experts: self.num_experts,
            num_experts_per_tok: self.num_experts_per_tok,
            norm_topk_prob: self.norm_topk_prob,
            quantized_weight_configs: None,
        }
    }
}

impl DecoderConfig {
    /// Returns the exact architecture identity established by `model_type`.
    pub fn architecture(&self) -> Architecture {
        match self.model_type.as_str() {
            "qwen2" => Architecture::Qwen2,
            "qwen3" | "qwen3_moe" | "qwen3_vl_text" | "qwen3_vl_moe_text" => Architecture::Qwen3,
            _ => unreachable!("validated dense-Qwen model type"),
        }
    }

    pub(crate) fn model_kind(&self) -> crate::core::ModelKind {
        match self.architecture() {
            Architecture::Qwen2 => crate::core::ModelKind::Qwen2,
            Architecture::Qwen3 => crate::core::ModelKind::Qwen3,
        }
    }

    /// Whether this architecture carries learned Q/K/V projection biases.
    pub fn qkv_bias(&self) -> bool {
        self.architecture() == Architecture::Qwen2
    }

    /// Whether this architecture applies per-head Q/K RMS normalization.
    pub fn qk_norm(&self) -> bool {
        self.architecture() == Architecture::Qwen3
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
        self.num_experts > 0
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
                        RopeValue::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        RopeValue::String(value) => format!("string:{value}"),
                        RopeValue::Bool(value) => format!("bool:{value}"),
                    };
                    format!("{key}={value}")
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    rope.sort_unstable();
    format!(
        "dense-qwen-v2:type={}:hidden={}:layers={}:q_heads={}:kv_heads={}:head_dim={}:context={}:rope_theta={:08x}:rope={}:attention={}",
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
    )
}

#[cfg(test)]
fn prompt_cache_layer_layout(
    args: &DecoderConfig,
) -> Result<LayerSchedule<crate::LayerCachePolicy>, Exception> {
    PromptCacheModelIdentity::key_value_layouts(
        args.attention_schedule.iter().map(|policy| {
            policy.window().map(|window| {
                i32::try_from(window.get()).expect("validated dense-Qwen window fits i32")
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
        .map_err(|_| Exception::custom("invalid dense-Qwen cache layer count"))?;
    Ok(PromptCacheModelIdentity {
        model_family: "dense_qwen".into(),
        effective_model_type: args.model_type.clone(),
        architecture_fingerprint: prompt_cache_architecture_fingerprint(args),
        layer_count,
        global_layer_start: 0,
        global_layer_end: layer_count,
        sink_tokens: 0,
        layer_prefix_offsets: vec![0; layer_count],
        topology: Default::default(),
        layer_layout: PromptCacheModelIdentity::key_value_layouts(
            args.attention_schedule.iter().map(|policy| {
                policy.window().map(|window| {
                    i32::try_from(window.get()).expect("validated dense-Qwen window fits i32")
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
        .map_err(|_| Exception::custom("invalid dense-Qwen cache layer count"))?;
    if cache.len() != layer_count {
        return Err(Exception::custom(format!(
            "dense-Qwen cache has {} layers, expected {layer_count}",
            cache.len()
        )));
    }
    let end = i64::try_from(prefix_token_ids.len())
        .map_err(|_| Exception::custom("dense-Qwen prompt length exceeds i64"))?;
    let mut blocks = Vec::with_capacity(layer_count);
    for (layer, cache) in cache.iter().enumerate() {
        let cache = cache.as_ref().ok_or_else(|| {
            Exception::custom(format!("dense-Qwen layer {layer} cache is missing"))
        })?;
        if i64::from(cache.offset()) != end {
            return Err(Exception::custom(format!(
                "dense-Qwen layer {layer} cache offset does not match the persisted prefix"
            )));
        }
        let (keys, values) = cache.snapshot_arrays(stream)?.ok_or_else(|| {
            Exception::custom(format!("dense-Qwen layer {layer} cache state is missing"))
        })?;
        let retained = i64::from(keys.dim(-2));
        blocks.push(PromptCacheSnapshotBlock {
            global_layer: layer,
            start: end.checked_sub(retained).ok_or_else(|| {
                Exception::custom(format!("dense-Qwen layer {layer} retained range underflow"))
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
            "dense-Qwen prompt cache contains unexpected fixed state",
        ));
    }
    let mut blocks = blocks
        .into_iter()
        .map(|block| (block.global_layer, block))
        .collect::<BTreeMap<_, _>>();
    let end = i32::try_from(prefix_token_ids.len())
        .map_err(|_| Exception::custom("dense-Qwen prompt length exceeds i32"))?;
    let mut cache = Vec::with_capacity(identity.layer_count);
    for layer in 0..identity.layer_count {
        let mut layer_cache = match args
            .attention_schedule
            .get(layer)
            .and_then(|policy| policy.window())
            .map(|window| {
                i32::try_from(window.get()).expect("validated dense-Qwen window fits i32")
            }) {
            Some(window) => ConcatKeyValueCache::new_for_sliding_attention(window),
            None => ConcatKeyValueCache::new(),
        };
        let block = blocks.remove(&layer).ok_or_else(|| {
            Exception::custom(format!(
                "dense-Qwen layer {layer} prompt-cache block is missing"
            ))
        })?;
        match block.arrays {
            CacheBlockArrays::KeyValue { keys, values } => {
                layer_cache.restore_resident(keys, values, end)?;
            }
            CacheBlockArrays::CompressedLatentRotary { .. } => {
                return Err(Exception::custom(format!(
                    "dense-Qwen layer {layer} prompt cache contains compressed latent state"
                )))
            }
        }
        cache.push(Some(layer_cache));
    }
    if !blocks.is_empty() {
        return Err(Exception::custom(
            "dense-Qwen prompt cache contains unexpected attention blocks",
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

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Shared dense-Qwen attention layer.
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
            stream,
        )
    }

    fn new_with_prefix(
        args: &DecoderConfig,
        prefix: &str,
        policy: AttentionPolicy,
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

        let q_norm = args
            .qk_norm()
            .then(|| nn::RmsNorm::unloaded(head_dim, args.rms_norm_eps, Dtype::Float32, stream))
            .transpose()?;
        let k_norm = args
            .qk_norm()
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
            q_norm,
            k_norm,
            rope,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(|_| Exception::custom("sliding attention window exceeds i32"))?,
        })
    }

    fn normalize_query(&mut self, value: Array, stream: &Stream) -> Result<Array, Exception> {
        match &mut self.q_norm {
            Some(norm) => norm.forward(&value, stream),
            None => Ok(value),
        }
    }

    fn normalize_key(&mut self, value: Array, stream: &Stream) -> Result<Array, Exception> {
        match &mut self.k_norm {
            Some(norm) => norm.forward(&value, stream),
            None => Ok(value),
        }
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
        if self.q_norm.is_some() {
            observer.observe(&format!("{prefix}.q_norm"), &queries)?;
        }
        let keys = self.normalize_key(
            reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?,
            stream,
        )?;
        if self.k_norm.is_some() {
            observer.observe(&format!("{prefix}.k_norm"), &keys)?;
        }
        let values = reshape_attention_projection(values, batch, seq_len, self.n_kv_heads, stream)?;
        observer.observe(&format!("{prefix}.values"), &values)?;

        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
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

        let output = self.o_proj.forward(&output, stream)?;
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
        let (queries, keys, values) = apply_rotary_embeddings_and_update_cache(
            queries, keys, values, cos, sin, &mut cache, stream,
        )?;
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
        self.o_proj.forward(&output, stream)
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
        let (queries, keys, values) = apply_rotary_embeddings_and_update_cache(
            queries, keys, values, cos, sin, &mut cache, stream,
        )?;
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
        crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut self.o_proj,
            &attended,
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
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
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

        self.o_proj.forward(&output, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        self.q_proj.training_mode(mode);
        self.k_proj.training_mode(mode);
        self.v_proj.training_mode(mode);
        self.o_proj.training_mode(mode);
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        let (indices, weights) = self.gate.forward(&flat, stream)?;
        let returned =
            crate::backend::mlx::architectures::distributed::expert::dispatch_replicated(
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
        prefix: &str,
        mut observer: Option<&mut dyn ActivationObserver>,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let shape = hidden_states.shape();
        let flat = hidden_states.reshape(&[-1, shape[2]], stream)?;
        crate::backend::mlx::architectures::distributed::expert::materialize_timing_phase([&flat])?;
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
        crate::backend::mlx::architectures::distributed::expert::materialize_timing_phase(
            router_outputs,
        )?;
        statistics.router_time += router_started.elapsed();
        let returned =
            crate::backend::mlx::architectures::distributed::expert::dispatch_replicated(
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
        crate::backend::mlx::architectures::distributed::expert::materialize_timing_phase([
            &output,
        ])?;
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
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
/// Shared dense-Qwen decoder block.
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
    /// Pre-MLP RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
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
            nn::RmsNorm::unloaded(args.hidden_size, args.rms_norm_eps, Dtype::Float32, stream)?;

        Ok(Self {
            num_attention_heads,
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
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
        let normed = self.input_layernorm.forward(x, stream)?;
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
        observer.observe(&format!("{prefix}.residual_delta_attention"), &r)?;
        let h = x.add(r, stream)?;
        observer.observe(&format!("{prefix}.post_attention_residual"), &h)?;
        observer.observe(&format!("{prefix}.residual_after_attention"), &h)?;

        let feed_forward_name = if self.mlp.is_moe() { "moe" } else { "mlp" };
        observer.observe(&format!("{prefix}.residual_before_{feed_forward_name}"), &h)?;
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &post_normed)?;
        let r = self.mlp.forward_with_observer(
            &post_normed,
            stream,
            &format!("{prefix}.mlp"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.{feed_forward_name}_output"), &r)?;
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
        let mlp = self.mlp.forward(&normed, stream)?;
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
        let attention = crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut attention.o_proj,
            &attended,
            group,
            stream,
        )?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate = crate::backend::mlx::nn::layers::silu(
                    mlp.gate_proj.forward(&normalized, stream)?,
                    stream,
                )?;
                let up = mlp.up_proj.forward(&normalized, stream)?;
                crate::backend::mlx::nn::parallel::forward_row_parallel(
                    &mut mlp.down_proj,
                    &gate.multiply(up, stream)?,
                    group,
                    stream,
                )?
            }
            FeedForward::Moe(moe) => moe.forward_tensor_parallel(&normalized, group, stream)?,
        };
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
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
        let attention = crate::backend::mlx::nn::parallel::forward_row_parallel(
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        tensor_group: &safemlx::distributed::Group,
        expert_group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
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
                group,
                stream,
            )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate = crate::backend::mlx::nn::layers::silu(
                    mlp.gate_proj.forward(&normed, stream)?,
                    stream,
                )?;
                let up = mlp.up_proj.forward(&normed, stream)?;
                crate::backend::mlx::nn::parallel::forward_row_parallel(
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
        let attention = crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut attention.o_proj,
            &attended,
            tensor_group,
            stream,
        )?;
        let hidden = x.add(attention, stream)?;
        let normed = self.post_attention_layernorm.forward(&hidden, stream)?;
        let feed_forward = match &mut self.mlp {
            FeedForward::Dense(mlp) => {
                let gate = crate::backend::mlx::nn::layers::silu(
                    mlp.gate_proj.forward(&normed, stream)?,
                    stream,
                )?;
                let up = mlp.up_proj.forward(&normed, stream)?;
                crate::backend::mlx::nn::parallel::forward_row_parallel(
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
                let gate = crate::backend::mlx::nn::layers::silu(
                    mlp.gate_proj.forward(&normed, stream)?,
                    stream,
                )?;
                let up = mlp.up_proj.forward(&normed, stream)?;
                crate::backend::mlx::nn::parallel::forward_row_parallel(
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
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
        assignment: &crate::backend::mlx::architectures::distributed::expert::ExpertAssignment,
        group: &safemlx::distributed::Group,
        statistics: &mut crate::backend::mlx::architectures::distributed::expert::RoutingStatistics,
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

        let normed = self.input_layernorm.forward(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
        };
        let r = self.self_attn.forward(self_attn_input, stream)?;
        let h = x.add(r, stream)?;

        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        let r = self.mlp.forward(&post_normed, stream)?;
        h.add(r, stream)
    }

    fn training_mode(&mut self, mode: bool) {
        <Attention as Module<AttentionInput<'_, C>>>::training_mode(&mut self.self_attn, mode);
        self.mlp.training_mode(mode);
        self.input_layernorm.training_mode(mode);
        self.post_attention_layernorm.training_mode(mode);
    }
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Shared dense-Qwen transformer body without the language-model head.
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
    /// Creates an unloaded dense-Qwen transformer body.
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
}

/// Input for a dense-Qwen forward pass.
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
    /// Creates an unloaded dense-Qwen causal language model.
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
    pub fn new_cache(
        &self,
    ) -> Vec<Option<crate::backend::mlx::runtime::cache::ConcatKeyValueCache>> {
        self.args
            .attention_schedule
            .iter()
            .map(|policy| {
                Some(match policy.window() {
                    Some(window) => {
                        crate::backend::mlx::runtime::cache::ConcatKeyValueCache::new_for_sliding_attention(
                            i32::try_from(window.get())
                                .expect("validated dense-Qwen attention window fits i32"),
                        )
                    }
                    None => crate::backend::mlx::runtime::cache::ConcatKeyValueCache::new(),
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
        observer.observe("lm_head.logits", &logits)?;
        Ok(logits)
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
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &out,
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

/// Loads `tokenizer.json` from a dense-Qwen model directory.
pub fn load_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads and validates a dense-Qwen decoder configuration from `config.json`.
pub fn load_config(model_dir: impl AsRef<Path>) -> Result<DecoderConfig, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let config: Value = serde_json::from_reader(file)?;
    config_from_hf_value(&config)
}

/// Parses and validates the arguments shared by structural preflight and loading.
pub(crate) fn config_from_hf_value(config: &Value) -> Result<DecoderConfig, Error> {
    if let Some(model_type) = config.get("model_type").and_then(Value::as_str) {
        validate_declared_architectures(config, model_type)?;
    }
    validate_execution_fields(config)?;
    let mut source =
        serde_json::from_value::<DecoderConfigSource>(config.clone()).map_err(|error| {
            Error::UnsupportedArchitecture(format!("invalid dense-Qwen config: {error}"))
        })?;
    source.normalize_head_dim();
    let attention_schedule =
        qwen_hf_attention_schedule(config, &source.model_type, source.num_hidden_layers)?;
    let args = source.into_config(attention_schedule);
    validate_model_args(&args)?;
    Ok(args)
}

pub(crate) fn qwen3_text_config_from_hf_value(
    config: &Value,
    model_type: &str,
) -> Result<DecoderConfig, Error> {
    let mut source =
        serde_json::from_value::<DecoderConfigSource>(config.clone()).map_err(|error| {
            Error::UnsupportedArchitecture(format!("invalid {model_type} text_config: {error}"))
        })?;
    source.model_type = model_type.into();
    source.normalize_head_dim();
    let layers = usize::try_from(source.num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "{model_type} num_hidden_layers must be positive, got {}",
            source.num_hidden_layers
        ))
    })?;
    let schedule = LayerSchedule::all_full(layers)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    Ok(source.into_config(schedule))
}

fn qwen_hf_attention_schedule(
    config: &Value,
    model_type: &str,
    num_hidden_layers: i32,
) -> Result<LayerSchedule<AttentionPolicy>, Error> {
    let layers = usize::try_from(num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "num_hidden_layers must be positive, got {}",
            num_hidden_layers
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "num_hidden_layers must be positive, got 0".into(),
        ));
    }
    let enabled = match config.get("use_sliding_window") {
        None => false,
        Some(value) => value.as_bool().ok_or_else(|| {
            Error::UnsupportedArchitecture("use_sliding_window must be boolean".into())
        })?,
    };
    if matches!(
        model_type,
        "qwen3" | "qwen3_moe" | "qwen3_vl_text" | "qwen3_vl_moe_text"
    ) {
        if enabled {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3 dense/MoE does not use Qwen2 sliding-window configuration".into(),
            ));
        }
        return LayerSchedule::all_full(layers)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
    }
    if model_type != "qwen2" {
        return Err(Error::UnsupportedModelType(model_type.into()));
    }
    if !enabled {
        return LayerSchedule::all_full(layers)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()));
    }

    let window = required_positive_hf_u32(config, "sliding_window", "use_sliding_window=true")?;
    if window > i32::MAX as u32 {
        return Err(Error::UnsupportedArchitecture(format!(
            "sliding_window exceeds the executable i32 range: {window}"
        )));
    }
    let first =
        required_nonnegative_hf_usize(config, "max_window_layers", "use_sliding_window=true")?;
    if first >= layers {
        return Err(Error::UnsupportedArchitecture(format!(
            "max_window_layers must select at least one configured layer, got {first} for {layers} layers"
        )));
    }
    let sliding = AttentionPolicy::sliding(window)
        .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    LayerSchedule::new(
        layers,
        (0..layers)
            .map(|layer| {
                if layer < first {
                    AttentionPolicy::Full
                } else {
                    sliding
                }
            })
            .collect(),
    )
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
        "layer_types",
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
                "dense-Qwen config field {field:?} changes decoder execution and is not supported"
            )));
        }
    }
    if let Some(value) = config.get("partial_rotary_factor") {
        match value.as_f64() {
            Some(1.0) => {}
            Some(_) => {
                return Err(Error::UnsupportedArchitecture(
                    "dense Qwen requires full-head rotary embeddings (partial_rotary_factor=1)"
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
                    "dense Qwen does not support interleaved RoPE coordinates".into(),
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
        "qwen2" => &["Qwen2ForCausalLM"],
        "qwen3" => &["Qwen3ForCausalLM"],
        "qwen3_moe" => &["Qwen3MoeForCausalLM"],
        _ => return Ok(()),
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
    if !matches!(
        args.model_type.as_str(),
        "qwen2" | "qwen3" | "qwen3_moe" | "qwen3_vl_text" | "qwen3_vl_moe_text"
    ) {
        return Err(Error::UnsupportedModelType(args.model_type.clone()));
    }
    if args.model_type == "qwen2" && args.is_moe() {
        return Err(Error::UnsupportedArchitecture(
            "model_type qwen2 supports only the dense Qwen2/Qwen2.5 text architecture".into(),
        ));
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
    let query_width = args
        .num_attention_heads
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
    // Qwen2 derives the per-head dimension from hidden_size, but Qwen3
    // checkpoints declare it independently. For example, Qwen3-0.6B uses a
    // 1024-wide residual stream and 16 x 128-wide query projections. The
    // attention implementation already projects hidden_size -> query_width
    // and query_width -> hidden_size, so only Qwen2 requires equality here.
    if args.architecture() == Architecture::Qwen2 && args.hidden_size != query_width {
        return Err(Error::UnsupportedArchitecture(format!(
            "hidden_size ({}) must equal num_attention_heads ({}) x head_dim ({})",
            args.hidden_size, args.num_attention_heads, args.head_dim
        )));
    }
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
            "dense Qwen requires hidden_act=\"silu\", got {:?}",
            args.hidden_act
        )));
    }
    if args.attention_dropout != 0.0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "dense Qwen inference requires attention_dropout=0, got {}",
            args.attention_dropout
        )));
    }
    match (args.architecture(), args.attention_bias) {
        (Architecture::Qwen2, Some(false)) => {
            return Err(Error::UnsupportedArchitecture(
                "Qwen2 text checkpoints require learned Q/K/V projection biases".into(),
            ));
        }
        (Architecture::Qwen3, Some(true)) => {
            return Err(Error::UnsupportedArchitecture(
                "Qwen3 dense attention does not support biased Q/K/V projections".into(),
            ));
        }
        _ => {}
    }
    if args.mlp_bias == Some(true) {
        return Err(Error::UnsupportedArchitecture(
            "dense Qwen does not support biased SwiGLU projections".into(),
        ));
    }
    validate_attention_schedule(args)?;
    validate_rope_scaling(args)?;
    if args.is_moe() {
        for (name, value) in [
            ("moe_intermediate_size", args.moe_intermediate_size),
            ("num_experts", args.num_experts),
            ("num_experts_per_tok", args.num_experts_per_tok),
        ] {
            if value <= 0 {
                return Err(Error::UnsupportedArchitecture(format!(
                    "{name} must be positive for Qwen3 MoE, got {value}"
                )));
            }
        }
        if args.num_experts_per_tok > args.num_experts {
            return Err(Error::UnsupportedArchitecture(format!(
                "num_experts_per_tok ({}) exceeds num_experts ({})",
                args.num_experts_per_tok, args.num_experts
            )));
        }
    } else if args.intermediate_size <= 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "intermediate_size must be positive for dense Qwen3, got {}",
            args.intermediate_size
        )));
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
    if args.architecture() == Architecture::Qwen3
        && args
            .attention_schedule
            .iter()
            .any(|policy| *policy != AttentionPolicy::Full)
    {
        return Err(Error::UnsupportedArchitecture(
            "Qwen3 dense/MoE requires full attention in every decoder layer".into(),
        ));
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
            RopeValue::String(value) => Some(value.as_str()),
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
            RopeValue::Float(value) if value.is_finite() => Some(*value),
            RopeValue::String(value) => value.parse::<f32>().ok().filter(|v| v.is_finite()),
            _ => None,
        })
    };
    if matches!(rope_type, "linear" | "yarn")
        && numeric("factor").is_none_or(|factor| factor <= 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "scaled dense-Qwen RoPE requires a finite positive factor".into(),
        ));
    }
    if rope_type == "yarn"
        && numeric("original_max_position_embeddings").is_none_or(|value| value <= 0.0)
    {
        return Err(Error::UnsupportedArchitecture(
            "YaRN dense-Qwen RoPE requires positive original_max_position_embeddings".into(),
        ));
    }
    Ok(())
}

pub(crate) fn prepare_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    is_moe: bool,
) -> Result<(DecoderConfig, Vec<u32>), Error> {
    let gguf_architecture = crate::core::GgufArchitecture::resolve(architecture)?;
    crate::backend::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
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
    let eos_token_ids = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
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

    let attention_schedule = if architecture == "qwen2" {
        qwen2_gguf_sliding_config(metadata, architecture, num_hidden_layers)?
    } else {
        LayerSchedule::all_full(num_hidden_layers as usize)
            .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?
    };
    let args = DecoderConfig {
        model_type: if architecture == "qwen2" {
            "qwen2"
        } else if is_moe {
            "qwen3_moe"
        } else {
            "qwen3"
        }
        .to_string(),
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
        vocab_size,
        num_key_value_heads,
        max_position_embeddings: gguf_i32_catalog(metadata, &key("context_length"))?,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or(1_000_000.0),
        head_dim,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        rope_scaling: gguf_rope_scaling(metadata, architecture)?,
        hidden_act: default_hidden_act(),
        attention_dropout: 0.0,
        attention_bias: (architecture == "qwen2").then_some(true),
        mlp_bias: Some(false),
        attention_schedule,
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
    };
    validate_model_args(&args)?;
    Ok(args)
}

fn qwen2_gguf_sliding_config(
    metadata: &HashMap<String, GgufMetadataValue>,
    architecture: &str,
    layers: i32,
) -> Result<LayerSchedule<AttentionPolicy>, Error> {
    let layers = usize::try_from(layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "Qwen2 GGUF decoder layer count must be positive, got {layers}"
        ))
    })?;
    if layers == 0 {
        return Err(Error::UnsupportedArchitecture(
            "Qwen2 GGUF decoder layer count must be positive, got 0".into(),
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
                "Qwen2 GGUF sliding-window pattern has {} entries for {layers} layers",
                pattern.len()
            )));
        }
    }
    let window = match window {
        Some(window) if window <= 0 => {
            return Err(Error::UnsupportedArchitecture(format!(
                "Qwen2 GGUF sliding window must be positive, got {window}"
            )))
        }
        Some(window) => Some(u32::try_from(window).map_err(|_| {
            Error::UnsupportedArchitecture(format!(
                "Qwen2 GGUF sliding window exceeds the u32 range: {window}"
            ))
        })?),
        None => None,
    };
    if window.is_some_and(|window| window > i32::MAX as u32) {
        return Err(Error::UnsupportedArchitecture(format!(
            "Qwen2 GGUF sliding window exceeds the executable i32 range: {}",
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
) -> Result<Option<HashMap<String, RopeValue>>, Error> {
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
                    RopeValue::String("linear".to_string()),
                ),
                ("factor".to_string(), RopeValue::Float(factor)),
            ])))
        }
        "yarn" => {
            // GGUF uses architecture-scoped names which differ from the
            // Hugging Face map consumed by `initialize_rope`.
            for suffix in ["yarn_ext_factor", "yarn_attn_factor", "yarn_log_multiplier"] {
                let key = format!("{architecture}.rope.scaling.{suffix}");
                if metadata.contains_key(&key) {
                    return Err(Error::UnsupportedArchitecture(format!(
                        "GGUF YaRN field {key:?} changes attention scaling semantics that the dense-Qwen decoder does not implement"
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
                    RopeValue::String("yarn".to_string()),
                ),
                ("factor".to_string(), RopeValue::Float(factor)),
                (
                    "original_max_position_embeddings".to_string(),
                    RopeValue::Float(original),
                ),
                ("beta_fast".to_string(), RopeValue::Float(beta_fast)),
                ("beta_slow".to_string(), RopeValue::Float(beta_slow)),
                ("truncate".to_string(), RopeValue::Bool(false)),
            ])))
        }
        other => Err(Error::UnsupportedArchitecture(format!(
            "GGUF RoPE scaling type {other:?} is not supported by the dense-Qwen GGUF loader"
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
        const MOE_PARAMETERS: [(&str, &str); 4] = [
            ("ffn_gate_inp", "mlp.gate"),
            ("ffn_gate_exps", "mlp.experts.gate_proj"),
            ("ffn_up_exps", "mlp.experts.up_proj"),
            ("ffn_down_exps", "mlp.experts.down_proj"),
        ];
        for (source, target) in MOE_PARAMETERS {
            if parameter == source || parameter.starts_with(&format!("{source}.")) {
                let mut suffix = parameter.strip_prefix(source).unwrap_or_default();
                if target.starts_with("mlp.experts.") {
                    suffix = match suffix {
                        ".weight" => "",
                        ".scales" => "_scales",
                        ".biases" => "_biases",
                        other => other,
                    };
                }
                return format!("model.layers.{layer}.{target}{suffix}");
            }
        }
    }

    const PARAMETERS: [(&str, &str); 12] = [
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
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
pub type Generate<'a, C, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, Model, Vec<Option<C>>, S>;
