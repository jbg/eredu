//! Validated backend-neutral Gemma 4 text configuration.

use std::{
    collections::{HashMap, HashSet},
    num::NonZeroU32,
    ops::Range,
};

use crate::{rotary::RopeValue, GgufTensorCatalog};
use eredu_checkpoint::{LinearFormat, WeightQuantization};
use eredu_core::{
    cache::derive_prompt_cache_architecture_fingerprint, AttentionPolicy, LayerSchedule,
};
use eredu_gguf::{MetadataArray, MetadataValue};
use eredu_nn::{AttentionStateSource, AttentionValueSource};
use serde::Deserialize;

/// Invalid or unsupported Gemma 4 configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// Configuration JSON could not be decoded.
    #[error("invalid Gemma 4 configuration: {0}")]
    Json(#[from] serde_json::Error),
    /// Geometry or scheduling changes unsupported semantics.
    #[error("{0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}

/// Feed-forward topology selected for one decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum FeedForwardPolicy {
    /// Dense gated MLP only.
    Dense,
    /// Dense gated MLP plus routed sparse experts.
    DenseWithSparseMoe,
}

/// Complete execution and mutable-state policy for one decoder layer.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub struct LayerPolicy {
    /// Full or exact-window sliding attention.
    pub attention: AttentionPolicy,
    /// Query/key/value head width.
    pub head_dim: NonZeroU32,
    /// Key/value head count.
    pub num_key_value_heads: NonZeroU32,
    /// State ownership, publication, and value projection topology.
    pub key_value: AttentionStateSource,
    /// Dense MLP intermediate width.
    pub intermediate_size: NonZeroU32,
    /// Dense-only or dense-plus-sparse topology.
    pub feed_forward: FeedForwardPolicy,
}

impl LayerPolicy {
    /// Stable schedule component used by prompt/cache identity.
    pub fn fingerprint_component(self) -> String {
        let attention = match self.attention {
            AttentionPolicy::Full => "f".to_owned(),
            AttentionPolicy::Sliding { window } => format!("s{}", window.get()),
        };
        let state = match self.key_value {
            AttentionStateSource::Local {
                value: AttentionValueSource::Projected,
            } => "lp",
            AttentionStateSource::Local {
                value: AttentionValueSource::ReuseKey,
            } => "lr",
            AttentionStateSource::Publish {
                value: AttentionValueSource::Projected,
            } => "pp",
            AttentionStateSource::Publish {
                value: AttentionValueSource::ReuseKey,
            } => "pr",
            AttentionStateSource::Shared => "s",
        };
        let feed_forward = match self.feed_forward {
            FeedForwardPolicy::Dense => "d",
            FeedForwardPolicy::DenseWithSparseMoe => "m",
        };
        format!(
            "{attention}:h{}:k{}:{state}:i{}:{feed_forward}",
            self.head_dim.get(),
            self.num_key_value_heads.get(),
            self.intermediate_size.get()
        )
    }
}

/// Normalized Gemma 4 decoder configuration.
#[derive(Debug, Clone)]
pub struct ModelArgs {
    /// Exact model identity (`gemma4` or `gemma4_unified`).
    pub model_type: String,
    /// Decoder hidden width.
    pub hidden_size: i32,
    /// Query head count.
    pub num_attention_heads: i32,
    /// Normalization epsilon.
    pub rms_norm_eps: f32,
    /// Token vocabulary size.
    pub vocab_size: i32,
    /// Media-safe padding token for per-layer embeddings.
    pub pad_token_id: i32,
    /// Maximum configured sequence length.
    pub max_position_embeddings: i32,
    /// Default rotary base.
    pub rope_theta: f32,
    /// Whether input and output token tables are tied.
    pub tie_word_embeddings: bool,
    /// Whether attention projections include bias.
    pub attention_bias: bool,
    /// Hidden width of optional per-layer token embeddings.
    pub hidden_size_per_layer_input: i32,
    /// Vocabulary size of optional per-layer token embeddings.
    pub vocab_size_per_layer_input: Option<i32>,
    /// Authoritative decoder layer schedule.
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    /// Optional final-logit soft cap.
    pub final_logit_softcapping: Option<f32>,
    /// Routed expert count.
    pub num_experts: Option<i32>,
    /// Selected experts per token.
    pub top_k_experts: Option<i32>,
    /// Routed expert intermediate width.
    pub moe_intermediate_size: Option<i32>,
    /// Default rotary scaling policy.
    pub rope_scaling: Option<HashMap<String, RopeValue>>,
    /// Per-attention-kind rotary overrides.
    pub rope_parameters: Option<HashMap<String, HashMap<String, RopeValue>>>,
    /// Preferred model-wide physical weight encoding.
    pub weight_quantization: Option<WeightQuantization>,
    /// Exact parameter names using the model-wide encoding in mixed artifacts.
    pub quantized_weights: Option<HashSet<String>>,
    /// Exact per-parameter encodings for mixed GGUF artifacts.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

impl ModelArgs {
    /// Parses and validates a Hugging Face text configuration document.
    pub fn from_hf_json(bytes: &[u8]) -> Result<Self, ConfigError> {
        normalize(serde_json::from_slice(bytes)?)
    }

    /// Parses strict Gemma 4 GGUF metadata and catalog-dependent storage
    /// policy without depending on a backend checkpoint type.
    pub fn from_gguf_metadata<C: GgufTensorCatalog + ?Sized>(
        catalog: &C,
        metadata: &HashMap<String, MetadataValue>,
    ) -> Result<Self, ConfigError> {
        let layers = gguf_i32(metadata, "gemma4.block_count")?;
        if layers <= 0 {
            return Err(invalid("Gemma 4 GGUF block_count must be positive"));
        }
        let count = layers as usize;
        let pattern = gguf_layer_pattern(metadata, count)?;
        let sliding_window = gguf_optional_i64(metadata, "gemma4.attention.sliding_window")?;
        if pattern.iter().any(|sliding| *sliding) && sliding_window.is_none_or(|window| window <= 0)
        {
            return Err(invalid(
                "Gemma 4 GGUF sliding layers require a positive sliding window",
            ));
        }
        let feed_forward = gguf_layer_i32_values(metadata, "gemma4.feed_forward_length", count)?;
        let kv_heads = gguf_layer_i32_values(metadata, "gemma4.attention.head_count_kv", count)?;
        let uniform = |name: &str, values: &[i32], select: &dyn Fn(usize) -> bool| {
            let selected = values
                .iter()
                .enumerate()
                .filter_map(|(layer, value)| select(layer).then_some(*value))
                .collect::<HashSet<_>>();
            if selected.len() > 1 {
                Err(invalid(format!(
                    "Gemma 4 GGUF {name} must be uniform within each attention policy"
                )))
            } else {
                Ok(selected.into_iter().next())
            }
        };
        let local_kv = uniform("KV heads", &kv_heads, &|layer| pattern[layer])?;
        let global_kv = uniform("KV heads", &kv_heads, &|layer| !pattern[layer])?;
        let shared = gguf_optional_i32(metadata, "gemma4.attention.shared_kv_layers")?.unwrap_or(0);
        if shared < 0 || shared > layers {
            return Err(invalid("Gemma 4 GGUF shared-KV layer count is invalid"));
        }
        let first_shared = (layers - shared) as usize;
        let mut key_as_value = None;
        for (layer, local) in pattern.iter().copied().enumerate().take(first_shared) {
            if local {
                continue;
            }
            let has_key = catalog.contains(&format!("blk.{layer}.attn_k.weight"));
            let reuses_key = has_key && !catalog.contains(&format!("blk.{layer}.attn_v.weight"));
            match key_as_value {
                Some(previous) if previous != reuses_key => {
                    return Err(invalid(
                        "Gemma 4 GGUF key-as-value policy differs between full-attention owners",
                    ))
                }
                None => key_as_value = Some(reuses_key),
                _ => {}
            }
        }
        let experts = gguf_optional_i32(metadata, "gemma4.expert_count")?.unwrap_or(0);
        let has_expert_weights = (0..count).any(|layer| {
            catalog.contains(&format!("blk.{layer}.ffn_gate_up_exps.weight"))
                || catalog.contains(&format!("blk.{layer}.ffn_gate_exps.weight"))
                || catalog.contains(&format!("blk.{layer}.ffn_down_exps.weight"))
        });
        let enable_moe = experts > 0 || has_expert_weights;
        let attention_bias = (0..count).any(|layer| {
            [
                "attn_q.bias",
                "attn_k.bias",
                "attn_v.bias",
                "attn_output.bias",
            ]
            .into_iter()
            .any(|local| catalog.contains(&format!("blk.{layer}.{local}")))
        });
        let vocabulary = gguf_vocab_size(metadata, "gemma4.vocab_size")?;
        let global_head = gguf_i32(metadata, "gemma4.attention.key_length")?;
        let local_head =
            gguf_optional_i32(metadata, "gemma4.attention.key_length_swa")?.unwrap_or(global_head);
        let full_rope =
            gguf_optional_f32(metadata, "gemma4.rope.freq_base")?.unwrap_or(1_000_000.0);
        let local_rope =
            gguf_optional_f32(metadata, "gemma4.rope.freq_base_swa")?.unwrap_or(10_000.0);
        let value = serde_json::json!({
            "model_type": "gemma4",
            "hidden_size": gguf_i32(metadata, "gemma4.embedding_length")?,
            "num_hidden_layers": layers,
            "intermediate_size": feed_forward[0],
            "feed_forward_lengths": feed_forward,
            "num_attention_heads": gguf_i32(metadata, "gemma4.attention.head_count")?,
            "rms_norm_eps": gguf_f32(metadata, "gemma4.attention.layer_norm_rms_epsilon")?,
            "vocab_size": vocabulary,
            "pad_token_id": 0,
            "num_key_value_heads": local_kv.or(global_kv).ok_or_else(|| invalid("Gemma 4 GGUF has no KV-head geometry"))?,
            "num_global_key_value_heads": global_kv,
            "max_position_embeddings": gguf_i32(metadata, "gemma4.context_length")?,
            "rope_theta": local_rope,
            "head_dim": local_head,
            "global_head_dim": global_head,
            "tie_word_embeddings": !catalog.contains("output.weight"),
            "attention_bias": attention_bias,
            "attention_k_eq_v": key_as_value.unwrap_or(false),
            "hidden_size_per_layer_input": gguf_optional_i32(metadata, "gemma4.embedding_length_per_layer_input")?.unwrap_or(0),
            "vocab_size_per_layer_input": vocabulary,
            "num_kv_shared_layers": shared,
            "layer_types": pattern.iter().map(|sliding| if *sliding { "sliding_attention" } else { "full_attention" }).collect::<Vec<_>>(),
            "sliding_window": sliding_window,
            "final_logit_softcapping": gguf_optional_f32(metadata, "gemma4.final_logit_softcapping")?,
            "enable_moe_block": enable_moe,
            "num_experts": enable_moe.then_some(experts),
            "top_k_experts": enable_moe.then(|| gguf_i32(metadata, "gemma4.expert_used_count")).transpose()?,
            "moe_intermediate_size": enable_moe.then(|| gguf_i32(metadata, "gemma4.expert_feed_forward_length")).transpose()?,
            "rope_parameters": {
                "full_attention": {"rope_type":"proportional", "partial_rotary_factor":0.25, "rope_theta":full_rope},
                "sliding_attention": {"rope_type":"default", "rope_theta":local_rope}
            }
        });
        Self::from_hf_json(&serde_json::to_vec(&value)?)
    }

    /// Number of decoder layers.
    pub fn num_hidden_layers(&self) -> usize {
        self.layer_schedule.len()
    }

    /// Balances decoder layers across pipeline stages without separating a
    /// shared-KV consumer from the layer that publishes its attention state.
    pub fn pipeline_layer_ranges(&self, stages: usize) -> Result<Vec<Range<usize>>, ConfigError> {
        let layers = self.num_hidden_layers();
        if stages == 0 || layers == 0 {
            return Err(invalid(
                "Gemma 4 pipeline planning requires positive layer and stage counts",
            ));
        }
        let mut can_split_after = vec![true; layers.saturating_sub(1)];
        let mut publishers = HashMap::new();
        for (layer, policy) in self.layer_schedule.iter().copied().enumerate() {
            match policy.key_value {
                AttentionStateSource::Publish { .. } => {
                    publishers.insert(policy.attention, layer);
                }
                AttentionStateSource::Shared => {
                    let publisher =
                        publishers.get(&policy.attention).copied().ok_or_else(|| {
                            invalid(format!(
                            "Gemma 4 layer {layer} consumes {:?} shared KV before any publisher",
                            policy.attention
                        ))
                        })?;
                    for boundary in can_split_after.iter_mut().take(layer).skip(publisher) {
                        *boundary = false;
                    }
                }
                AttentionStateSource::Local { .. } => {}
            }
        }

        let mut units = Vec::new();
        let mut start = 0;
        for (boundary, can_split) in can_split_after.iter().copied().enumerate() {
            if can_split {
                units.push(start..boundary + 1);
                start = boundary + 1;
            }
        }
        units.push(start..layers);
        if units.len() < stages {
            return Err(invalid(format!(
                "{stages} pipeline stages cannot be assigned to {} dependency-safe Gemma 4 decoder units; reduce pipeline_parallel_size",
                units.len()
            )));
        }

        let count = units.len();
        let mut prefix = vec![0usize; count + 1];
        for (index, unit) in units.iter().enumerate() {
            prefix[index + 1] = prefix[index] + unit.len();
        }
        let mut cost = vec![vec![usize::MAX; count + 1]; stages + 1];
        let mut split = vec![vec![0usize; count + 1]; stages + 1];
        cost[0][0] = 0;
        for groups in 1..=stages {
            for end in groups..=count {
                for previous in groups - 1..end {
                    let candidate = cost[groups - 1][previous].max(prefix[end] - prefix[previous]);
                    if candidate < cost[groups][end] {
                        cost[groups][end] = candidate;
                        split[groups][end] = previous;
                    }
                }
            }
        }
        let mut cuts = vec![count];
        let mut end = count;
        for groups in (1..=stages).rev() {
            end = split[groups][end];
            cuts.push(end);
        }
        cuts.reverse();
        Ok(cuts
            .windows(2)
            .map(|cut| units[cut[0]].start..units[cut[1] - 1].end)
            .collect())
    }

    /// Returns one complete layer policy without fallback.
    pub fn layer_policy(&self, layer: usize) -> Option<LayerPolicy> {
        self.layer_schedule.get(layer).copied()
    }

    /// Returns the rotary base selected for one attention kind.
    pub fn rope_theta_for(&self, attention: AttentionPolicy) -> f32 {
        let key = attention_key(attention);
        self.rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get(key))
            .and_then(|parameters| parameters.get("rope_theta"))
            .and_then(rope_float)
            .unwrap_or(self.rope_theta)
    }

    /// Returns normalized rotary metadata selected for one attention kind.
    pub fn rope_scaling_for(
        &self,
        attention: AttentionPolicy,
    ) -> Option<&HashMap<String, RopeValue>> {
        self.rope_parameters
            .as_ref()
            .and_then(|parameters| parameters.get(attention_key(attention)))
            .or(self.rope_scaling.as_ref())
    }

    /// Stable architecture/cache identity derived entirely from normalized policy.
    pub fn architecture_fingerprint(&self) -> String {
        let layer_rope = self
            .layer_schedule
            .iter()
            .map(|layer| {
                let mut scaling = self
                    .rope_parameters
                    .as_ref()
                    .and_then(|parameters| parameters.get(attention_key(layer.attention)))
                    .cloned()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|(key, value)| format!("{key}={value:?}"))
                    .collect::<Vec<_>>();
                scaling.sort_unstable();
                format!(
                    "{:?}:{:08x}:{}",
                    layer.attention,
                    self.rope_theta_for(layer.attention).to_bits(),
                    scaling.join(";")
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        derive_prompt_cache_architecture_fingerprint(
            "gemma4",
            [
                ("model_type", self.model_type.clone()),
                ("hidden_size", self.hidden_size.to_string()),
                ("num_hidden_layers", self.num_hidden_layers().to_string()),
                ("num_attention_heads", self.num_attention_heads.to_string()),
                (
                    "max_position_embeddings",
                    self.max_position_embeddings.to_string(),
                ),
                (
                    "layer_schedule",
                    self.layer_schedule
                        .iter()
                        .map(|policy| policy.fingerprint_component())
                        .collect::<Vec<_>>()
                        .join(","),
                ),
                ("layer_rope", layer_rope),
                ("quantization", format!("{:?}", self.weight_quantization)),
                (
                    "quantized_weights",
                    crate::cache_identity::string_set(self.quantized_weights.as_ref()),
                ),
                (
                    "quantized_weight_configs",
                    crate::cache_identity::debug_map(self.quantized_weight_configs.as_ref()),
                ),
            ],
        )
    }

    /// Returns the physical encoding for one canonical linear weight.
    pub fn linear_format_for(&self, name: &str) -> LinearFormat {
        if let Some(format) = self
            .quantized_weight_configs
            .as_ref()
            .and_then(|formats| formats.get(name))
        {
            return (*format).into();
        }
        match self.weight_quantization {
            Some(format)
                if self
                    .quantized_weights
                    .as_ref()
                    .is_none_or(|weights| weights.contains(name)) =>
            {
                format.into()
            }
            _ => LinearFormat::Dense,
        }
    }
}

fn attention_key(attention: AttentionPolicy) -> &'static str {
    match attention {
        AttentionPolicy::Full => "full_attention",
        AttentionPolicy::Sliding { .. } => "sliding_attention",
    }
}

fn rope_float(value: &RopeValue) -> Option<f32> {
    match value {
        RopeValue::Float(value) => Some(*value),
        RopeValue::String(value) => value.parse().ok(),
        RopeValue::Bool(_) => None,
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AttentionKindSource {
    SlidingAttention,
    FullAttention,
}

#[derive(Debug, Deserialize)]
struct ModelArgsSource {
    #[serde(default = "default_model_type")]
    model_type: String,
    hidden_size: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    #[serde(default)]
    use_double_wide_mlp: bool,
    #[serde(default)]
    feed_forward_lengths: Option<Vec<i32>>,
    num_attention_heads: i32,
    rms_norm_eps: f32,
    vocab_size: i32,
    #[serde(default)]
    pad_token_id: i32,
    num_key_value_heads: i32,
    #[serde(default)]
    num_global_key_value_heads: Option<i32>,
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    head_dim: i32,
    #[serde(default)]
    global_head_dim: Option<i32>,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    attention_k_eq_v: bool,
    #[serde(default)]
    hidden_size_per_layer_input: i32,
    #[serde(default)]
    vocab_size_per_layer_input: Option<i32>,
    #[serde(default)]
    num_kv_shared_layers: i32,
    #[serde(default)]
    layer_types: Vec<AttentionKindSource>,
    #[serde(default)]
    sliding_window: Option<i64>,
    #[serde(default)]
    final_logit_softcapping: Option<f32>,
    #[serde(default)]
    enable_moe_block: bool,
    #[serde(default)]
    num_experts: Option<i32>,
    #[serde(default)]
    top_k_experts: Option<i32>,
    #[serde(default, alias = "expert_intermediate_size")]
    moe_intermediate_size: Option<i32>,
    #[serde(default)]
    rope_scaling: Option<HashMap<String, RopeValue>>,
    #[serde(default)]
    rope_parameters: Option<HashMap<String, HashMap<String, RopeValue>>>,
    #[serde(default)]
    weight_quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantized_weights: Option<HashSet<String>>,
    #[serde(default)]
    quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

fn default_model_type() -> String {
    "gemma4".into()
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_true() -> bool {
    true
}

fn normalize(source: ModelArgsSource) -> Result<ModelArgs, ConfigError> {
    if !matches!(source.model_type.as_str(), "gemma4" | "gemma4_unified") {
        return Err(invalid(format!(
            "unsupported Gemma 4 model_type {:?}",
            source.model_type
        )));
    }
    let layers = usize::try_from(source.num_hidden_layers)
        .ok()
        .filter(|layers| *layers > 0)
        .ok_or_else(|| invalid("Gemma 4 num_hidden_layers must be positive"))?;
    for (name, value) in [
        ("hidden_size", source.hidden_size),
        ("intermediate_size", source.intermediate_size),
        ("num_attention_heads", source.num_attention_heads),
        ("num_key_value_heads", source.num_key_value_heads),
        ("head_dim", source.head_dim),
        ("vocab_size", source.vocab_size),
        ("max_position_embeddings", source.max_position_embeddings),
    ] {
        if value <= 0 {
            return Err(invalid(format!("Gemma 4 {name} must be positive")));
        }
    }
    if !source.rms_norm_eps.is_finite()
        || source.rms_norm_eps <= 0.0
        || !source.rope_theta.is_finite()
        || source.rope_theta <= 0.0
        || source
            .final_logit_softcapping
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(invalid(
            "Gemma 4 epsilon, rotary base, and logit cap must be finite and positive",
        ));
    }
    let attention = attention_schedule(&source, layers)?;
    let first_shared = layers
        .checked_sub(
            usize::try_from(source.num_kv_shared_layers)
                .map_err(|_| invalid("Gemma 4 shared-KV layer count must be non-negative"))?,
        )
        .ok_or_else(|| invalid("Gemma 4 shared-KV layer count exceeds decoder depth"))?;
    let feed_forward_lengths = source.feed_forward_lengths.clone().unwrap_or_else(|| {
        (0..layers)
            .map(|layer| {
                if source.use_double_wide_mlp && layer >= first_shared {
                    source.intermediate_size.saturating_mul(2)
                } else {
                    source.intermediate_size
                }
            })
            .collect()
    });
    if feed_forward_lengths.len() != layers {
        return Err(invalid(format!(
            "Gemma 4 feed-forward widths has {} entries for {layers} layers",
            feed_forward_lengths.len()
        )));
    }
    if source.use_double_wide_mlp
        && source.intermediate_size.checked_mul(2).is_none()
        && first_shared < layers
    {
        return Err(invalid("Gemma 4 doubled MLP width exceeds i32"));
    }
    let kv_heads = attention
        .iter()
        .map(|policy| {
            if *policy == AttentionPolicy::Full {
                source
                    .num_global_key_value_heads
                    .unwrap_or(source.num_key_value_heads)
            } else {
                source.num_key_value_heads
            }
        })
        .collect::<Vec<_>>();
    let head_dims = attention
        .iter()
        .map(|policy| {
            if *policy == AttentionPolicy::Full {
                source.global_head_dim.unwrap_or(source.head_dim)
            } else {
                source.head_dim
            }
        })
        .collect::<Vec<_>>();
    let values = attention
        .iter()
        .map(|policy| {
            if source.attention_k_eq_v && *policy == AttentionPolicy::Full {
                AttentionValueSource::ReuseKey
            } else {
                AttentionValueSource::Projected
            }
        })
        .collect::<Vec<_>>();
    let layer_schedule = layer_schedule(
        &attention,
        &feed_forward_lengths,
        &kv_heads,
        &head_dims,
        first_shared,
        &values,
        source.enable_moe_block,
    )?;
    validate_moe(&source)?;
    crate::rotary::normalize_algorithm(source.rope_scaling.as_ref()).map_err(invalid)?;
    if let Some(parameters) = &source.rope_parameters {
        for values in parameters.values() {
            crate::rotary::normalize_algorithm(Some(values)).map_err(invalid)?;
        }
    }
    Ok(ModelArgs {
        model_type: source.model_type,
        hidden_size: source.hidden_size,
        num_attention_heads: source.num_attention_heads,
        rms_norm_eps: source.rms_norm_eps,
        vocab_size: source.vocab_size,
        pad_token_id: source.pad_token_id,
        max_position_embeddings: source.max_position_embeddings,
        rope_theta: source.rope_theta,
        tie_word_embeddings: source.tie_word_embeddings,
        attention_bias: source.attention_bias,
        hidden_size_per_layer_input: source.hidden_size_per_layer_input,
        vocab_size_per_layer_input: source.vocab_size_per_layer_input,
        layer_schedule,
        final_logit_softcapping: source.final_logit_softcapping,
        num_experts: source.num_experts,
        top_k_experts: source.top_k_experts,
        moe_intermediate_size: source.moe_intermediate_size,
        rope_scaling: source.rope_scaling,
        rope_parameters: source.rope_parameters,
        weight_quantization: source.weight_quantization,
        quantized_weights: source.quantized_weights,
        quantized_weight_configs: source.quantized_weight_configs,
    })
}

fn attention_schedule(
    source: &ModelArgsSource,
    layers: usize,
) -> Result<LayerSchedule<AttentionPolicy>, ConfigError> {
    if !source.layer_types.is_empty() && source.layer_types.len() != layers {
        return Err(invalid(format!(
            "Gemma 4 layer_types has {} entries for {layers} layers",
            source.layer_types.len()
        )));
    }
    let pattern = if source.layer_types.is_empty() {
        vec![AttentionKindSource::FullAttention; layers]
    } else {
        source.layer_types.clone()
    };
    let sliding = source
        .sliding_window
        .map(|window| {
            let window = u32::try_from(window)
                .ok()
                .filter(|window| *window <= i32::MAX as u32)
                .ok_or_else(|| invalid("Gemma 4 sliding_window must be positive and fit i32"))?;
            AttentionPolicy::sliding(window).map_err(|error| invalid(error.to_string()))
        })
        .transpose()?;
    let policies = pattern
        .into_iter()
        .map(|kind| match kind {
            AttentionKindSource::FullAttention => Ok(AttentionPolicy::Full),
            AttentionKindSource::SlidingAttention => {
                sliding.ok_or_else(|| invalid("Gemma 4 sliding layer requires sliding_window"))
            }
        })
        .collect::<Result<Vec<_>, _>>()?;
    LayerSchedule::new(layers, policies).map_err(|error| invalid(error.to_string()))
}

fn layer_schedule(
    attention: &LayerSchedule<AttentionPolicy>,
    feed_forward: &[i32],
    kv_heads: &[i32],
    head_dims: &[i32],
    first_shared: usize,
    values: &[AttentionValueSource],
    enable_moe: bool,
) -> Result<LayerSchedule<LayerPolicy>, ConfigError> {
    let layers = attention.len();
    let shared_attention = attention
        .iter()
        .skip(first_shared)
        .copied()
        .collect::<HashSet<_>>();
    let publishers = shared_attention
        .into_iter()
        .filter_map(|required| {
            attention
                .iter()
                .take(first_shared)
                .enumerate()
                .filter_map(|(layer, candidate)| (*candidate == required).then_some(layer))
                .last()
        })
        .collect::<HashSet<_>>();
    for layer in first_shared..layers {
        let layer_attention = *attention
            .get(layer)
            .expect("validated Gemma 4 layer schedule");
        if let Some(provider) = attention
            .iter()
            .take(first_shared)
            .enumerate()
            .filter_map(|(index, candidate)| (*candidate == layer_attention).then_some(index))
            .last()
        {
            if kv_heads[provider] != kv_heads[layer] || head_dims[provider] != head_dims[layer] {
                return Err(invalid(format!(
                    "Gemma 4 shared-KV layer {layer} geometry does not match publisher {provider}"
                )));
            }
        } else {
            return Err(invalid(format!(
                "Gemma 4 shared-KV layer {layer} has no earlier matching attention publisher"
            )));
        }
    }
    let positive = |name: &str, layer: usize, value: i32| {
        u32::try_from(value)
            .ok()
            .and_then(NonZeroU32::new)
            .ok_or_else(|| invalid(format!("Gemma 4 layer {layer} {name} must be positive")))
    };
    let head_dimension = |layer: usize, value: i32| {
        u32::try_from(value)
            .ok()
            .filter(|value| *value >= 2)
            .and_then(NonZeroU32::new)
            .ok_or_else(|| {
                invalid(format!(
                    "Gemma 4 layer {layer} head dimension must be at least 2"
                ))
            })
    };
    let policies = (0..layers)
        .map(|layer| {
            let attention = *attention
                .get(layer)
                .expect("validated Gemma 4 layer schedule");
            Ok(LayerPolicy {
                attention,
                head_dim: head_dimension(layer, head_dims[layer])?,
                num_key_value_heads: positive("KV heads", layer, kv_heads[layer])?,
                key_value: if layer >= first_shared {
                    AttentionStateSource::Shared
                } else if publishers.contains(&layer) {
                    AttentionStateSource::Publish {
                        value: values[layer],
                    }
                } else {
                    AttentionStateSource::Local {
                        value: values[layer],
                    }
                },
                intermediate_size: positive("feed-forward width", layer, feed_forward[layer])?,
                feed_forward: if enable_moe {
                    FeedForwardPolicy::DenseWithSparseMoe
                } else {
                    FeedForwardPolicy::Dense
                },
            })
        })
        .collect::<Result<Vec<_>, ConfigError>>()?;
    LayerSchedule::new(layers, policies).map_err(|error| invalid(error.to_string()))
}

fn validate_moe(source: &ModelArgsSource) -> Result<(), ConfigError> {
    if source.enable_moe_block {
        let experts = source.num_experts.unwrap_or(0);
        let top_k = source.top_k_experts.unwrap_or(0);
        let intermediate = source.moe_intermediate_size.unwrap_or(0);
        if experts <= 0 || top_k <= 0 || top_k > experts || intermediate <= 0 {
            return Err(invalid(
                "Gemma 4 MoE requires positive expert count/width and top_k <= experts",
            ));
        }
    } else if source.num_experts.is_some()
        || source.top_k_experts.is_some()
        || source.moe_intermediate_size.is_some()
    {
        return Err(invalid(
            "Gemma 4 dense config cannot declare sparse expert geometry",
        ));
    }
    Ok(())
}

fn gguf_layer_pattern(
    metadata: &HashMap<String, MetadataValue>,
    layers: usize,
) -> Result<Vec<bool>, ConfigError> {
    let key = "gemma4.attention.sliding_window_pattern";
    let values = match metadata.get(key) {
        None => return Ok(vec![false; layers]),
        Some(MetadataValue::Array(MetadataArray::Bool(values))) => values.clone(),
        Some(value) => value
            .to_i64_vec()
            .ok_or_else(|| {
                invalid(format!(
                    "Gemma 4 GGUF {key:?} must be an integer/bool array"
                ))
            })?
            .into_iter()
            .map(|value| value != 0)
            .collect(),
    };
    if values.len() != layers {
        return Err(invalid(format!(
            "Gemma 4 GGUF sliding pattern has {} entries for {layers} layers",
            values.len()
        )));
    }
    Ok(values)
}

fn gguf_layer_i32_values(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
    layers: usize,
) -> Result<Vec<i32>, ConfigError> {
    let values = metadata
        .get(key)
        .and_then(MetadataValue::to_i64_vec)
        .ok_or_else(|| invalid(format!("Gemma 4 GGUF is missing integer array {key:?}")))?;
    let values = if values.len() == 1 {
        vec![values[0]; layers]
    } else if values.len() == layers {
        values
    } else {
        return Err(invalid(format!(
            "Gemma 4 GGUF {key:?} has {} entries for {layers} layers",
            values.len()
        )));
    };
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| invalid(format!("Gemma 4 GGUF {key:?} values must be positive i32")))
        })
        .collect()
}

fn gguf_vocab_size(
    metadata: &HashMap<String, MetadataValue>,
    fallback: &str,
) -> Result<i32, ConfigError> {
    gguf_i32(metadata, fallback)
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    gguf_optional_i32(metadata, key)?
        .ok_or_else(|| invalid(format!("Gemma 4 GGUF is missing {key:?}")))
}

fn gguf_optional_i32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i32>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .and_then(|value| i32::try_from(value).ok())
                .ok_or_else(|| invalid(format!("Gemma 4 GGUF {key:?} must be an i32")))
        })
        .transpose()
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_i64()
                .ok_or_else(|| invalid(format!("Gemma 4 GGUF {key:?} must be an integer")))
        })
        .transpose()
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("Gemma 4 GGUF is missing {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    metadata
        .get(key)
        .map(|value| {
            value
                .as_f32()
                .ok_or_else(|| invalid(format!("Gemma 4 GGUF {key:?} must be numeric")))
        })
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture() -> serde_json::Value {
        serde_json::json!({
            "model_type": "gemma4_unified",
            "hidden_size": 32,
            "num_hidden_layers": 6,
            "intermediate_size": 64,
            "use_double_wide_mlp": true,
            "num_attention_heads": 4,
            "rms_norm_eps": 0.000001,
            "vocab_size": 128,
            "pad_token_id": 0,
            "num_key_value_heads": 2,
            "num_global_key_value_heads": 1,
            "max_position_embeddings": 256,
            "rope_theta": 10000.0,
            "head_dim": 8,
            "global_head_dim": 16,
            "attention_k_eq_v": true,
            "num_kv_shared_layers": 2,
            "layer_types": [
                "sliding_attention", "full_attention", "sliding_attention",
                "full_attention", "sliding_attention", "full_attention"
            ],
            "sliding_window": 32,
            "final_logit_softcapping": 30.0,
            "rope_parameters": {
                "full_attention": { "rope_theta": 1000000.0 },
                "sliding_attention": { "rope_theta": 10000.0 }
            }
        })
    }

    #[test]
    fn normalizes_exact_attention_geometry_shared_kv_and_dense_widths() {
        let args = ModelArgs::from_hf_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        assert_eq!(args.num_hidden_layers(), 6);
        assert_eq!(args.layer_policy(0).unwrap().head_dim.get(), 8);
        assert_eq!(args.layer_policy(1).unwrap().head_dim.get(), 16);
        assert_eq!(
            args.layer_policy(3).unwrap().key_value,
            AttentionStateSource::Publish {
                value: AttentionValueSource::ReuseKey
            }
        );
        assert_eq!(
            args.layer_policy(4).unwrap().key_value,
            AttentionStateSource::Shared
        );
        assert_eq!(args.layer_policy(4).unwrap().intermediate_size.get(), 128);
        assert_eq!(args.rope_theta_for(AttentionPolicy::Full), 1_000_000.0);
        assert_ne!(args.architecture_fingerprint(), "");
    }

    #[test]
    fn prompt_cache_fingerprint_includes_effective_quantization() {
        let mut args = ModelArgs::from_hf_json(&serde_json::to_vec(&fixture()).unwrap()).unwrap();
        let dense = args.architecture_fingerprint();
        args.weight_quantization = Some(
            eredu_checkpoint::AffineQuantization::new(16, 4)
                .unwrap()
                .into(),
        );

        assert_ne!(dense, args.architecture_fingerprint());
    }

    #[test]
    fn rejects_schedule_shared_geometry_and_moe_drift() {
        let mut value = fixture();
        value["layer_types"] = serde_json::json!(["full_attention"]);
        assert!(ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());

        let mut value = fixture();
        value["num_global_key_value_heads"] = serde_json::json!(2);
        value["global_head_dim"] = serde_json::json!(8);
        // Matching publishers and consumers remain valid after a global geometry change.
        assert!(ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_ok());

        let mut value = fixture();
        value["enable_moe_block"] = serde_json::json!(true);
        value["num_experts"] = serde_json::json!(4);
        value["top_k_experts"] = serde_json::json!(5);
        value["moe_intermediate_size"] = serde_json::json!(16);
        assert!(ModelArgs::from_hf_json(&serde_json::to_vec(&value).unwrap()).is_err());
    }

    #[test]
    fn rejects_head_widths_without_a_rotary_pair() {
        let mut local = fixture();
        local["head_dim"] = serde_json::json!(1);
        let error = ModelArgs::from_hf_json(&serde_json::to_vec(&local).unwrap()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Gemma 4 layer 0 head dimension must be at least 2"
        );

        let mut global = fixture();
        global["global_head_dim"] = serde_json::json!(1);
        let error = ModelArgs::from_hf_json(&serde_json::to_vec(&global).unwrap()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "Gemma 4 layer 1 head dimension must be at least 2"
        );
    }

    #[test]
    fn parses_portable_gguf_schedule_shared_kv_and_key_as_value() {
        let mut metadata = HashMap::from([
            ("gemma4.block_count".into(), MetadataValue::Uint32(4)),
            ("gemma4.embedding_length".into(), MetadataValue::Uint32(32)),
            (
                "gemma4.attention.head_count".into(),
                MetadataValue::Uint32(4),
            ),
            (
                "gemma4.attention.key_length".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "gemma4.attention.key_length_swa".into(),
                MetadataValue::Uint32(8),
            ),
            (
                "gemma4.attention.shared_kv_layers".into(),
                MetadataValue::Uint32(2),
            ),
            (
                "gemma4.attention.sliding_window".into(),
                MetadataValue::Uint32(16),
            ),
            (
                "gemma4.attention.layer_norm_rms_epsilon".into(),
                MetadataValue::Float32(1e-6),
            ),
            ("gemma4.vocab_size".into(), MetadataValue::Uint32(64)),
            ("gemma4.context_length".into(), MetadataValue::Uint32(128)),
            (
                "gemma4.final_logit_softcapping".into(),
                MetadataValue::Float32(30.0),
            ),
            (
                "gemma4.feed_forward_length".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![64])),
            ),
            (
                "gemma4.attention.head_count_kv".into(),
                MetadataValue::Array(MetadataArray::Uint32(vec![1, 2, 1, 2])),
            ),
            (
                "gemma4.attention.sliding_window_pattern".into(),
                MetadataValue::Array(MetadataArray::Bool(vec![true, false, true, false])),
            ),
        ]);
        metadata.insert(
            "tokenizer.ggml.padding_token_id".into(),
            MetadataValue::String("wrong type".into()),
        );
        let catalog = HashSet::from([
            "output.weight".into(),
            "blk.1.attn_k.weight".into(),
            "blk.0.attn_k.weight".into(),
            "blk.0.attn_v.weight".into(),
        ]);
        let args = ModelArgs::from_gguf_metadata(&catalog, &metadata).unwrap();
        assert_eq!(args.pad_token_id, 0);
        assert!(!args.tie_word_embeddings);
        assert_eq!(args.num_hidden_layers(), 4);
        assert!(matches!(
            args.layer_policy(1).unwrap().key_value,
            AttentionStateSource::Publish {
                value: AttentionValueSource::ReuseKey
            }
        ));
        assert_eq!(
            args.layer_policy(3).unwrap().key_value,
            AttentionStateSource::Shared
        );
    }
}
