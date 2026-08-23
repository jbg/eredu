//! Strict backend-independent Nemotron-H configuration normalization.

use std::{
    collections::{HashMap, HashSet},
    io::Read,
    num::NonZeroU32,
};

use eredu_checkpoint::{AffineQuantization, WeightQuantization};
use eredu_core::{
    cache::{
        derive_prompt_cache_architecture_fingerprint, LayerCachePolicy, MutableStateResidency,
        StateTensorDimension, StateTensorDtype, StateTensorPolicy, StateTensorRole,
    },
    AttentionPolicy, LayerSchedule,
};
use eredu_gguf::MetadataValue;
use eredu_runtime::{StateLayout, StateSegmentLifetime, StateSegmentSpec};
use serde::Deserialize;
use serde_json::Value;

/// Stable segment identity for target decoder state.
pub const TARGET_STATE_SEGMENT: &str = "target";
/// Stable segment identity for checkpoint-embedded prediction state.
pub const PREDICTION_STATE_SEGMENT: &str = "prediction";

/// Executable physical unit at one Nemotron-H schedule position.
#[derive(Debug, Clone, Copy, Eq, Hash, PartialEq)]
pub enum LayerPolicy {
    /// Mamba2 state-space unit.
    Mamba,
    /// Grouped-query attention unit.
    SelfAttention(AttentionPolicy),
    /// Dense squared-ReLU MLP unit.
    DenseMlp,
    /// Grouped routed plus shared expert unit.
    SparseMoe,
}

impl LayerPolicy {
    fn from_marker(marker: char, attention: AttentionPolicy) -> Result<Self, ConfigError> {
        match marker {
            'M' => Ok(Self::Mamba),
            '*' => Ok(Self::SelfAttention(attention)),
            '-' => Ok(Self::DenseMlp),
            'E' => Ok(Self::SparseMoe),
            marker => Err(invalid(format!(
                "hybrid_override_pattern contains unsupported marker {marker:?}"
            ))),
        }
    }
}

/// Backend-neutral floating-point storage policy.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum WeightDtype {
    /// IEEE float32.
    Float32,
    /// IEEE float16.
    Float16,
    /// Brain float16.
    Bfloat16,
}

/// Rank-local state geometry resolved from semantic placement.
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(missing_docs)]
pub enum LayerGeometry {
    Mamba { heads: i32, groups: i32 },
    Attention { query_heads: i32, kv_heads: i32 },
    DenseMlp { intermediate: i32 },
    SparseMoe { routed: i32, shared: i32 },
}

#[derive(Debug, Clone, Deserialize)]
struct Source {
    model_type: String,
    #[serde(default = "default_vocab_size")]
    vocab_size: i32,
    #[serde(default)]
    tie_word_embeddings: bool,
    #[serde(default = "default_hidden_size")]
    hidden_size: i32,
    #[serde(default = "default_intermediate_size")]
    intermediate_size: i32,
    #[serde(default = "default_num_hidden_layers")]
    num_hidden_layers: i32,
    #[serde(default)]
    num_nextn_predict_layers: i32,
    #[serde(default)]
    mtp_hybrid_override_pattern: Option<String>,
    #[serde(default)]
    mtp_layers_block_type: Option<Vec<String>>,
    #[serde(default = "default_hybrid_override_pattern")]
    hybrid_override_pattern: String,
    #[serde(default = "default_num_attention_heads")]
    num_attention_heads: i32,
    #[serde(default = "default_head_dim")]
    head_dim: i32,
    #[serde(default = "default_num_key_value_heads")]
    num_key_value_heads: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default = "default_max_position_embeddings")]
    max_position_embeddings: i32,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    #[serde(default)]
    use_bias: bool,
    #[serde(default = "default_norm_eps")]
    layer_norm_epsilon: f32,
    #[serde(default = "default_norm_eps")]
    norm_eps: f32,
    #[serde(default)]
    residual_in_fp32: bool,
    #[serde(default = "default_num_logits_to_keep")]
    num_logits_to_keep: i32,
    #[serde(default)]
    sliding_window: Option<i64>,
    #[serde(default = "default_ssm_state_size")]
    ssm_state_size: i32,
    #[serde(default = "default_mamba_num_heads")]
    mamba_num_heads: i32,
    #[serde(default = "default_n_groups")]
    n_groups: i32,
    #[serde(default = "default_mamba_head_dim")]
    mamba_head_dim: i32,
    #[serde(default = "default_conv_kernel")]
    conv_kernel: i32,
    #[serde(default = "default_expand")]
    expand: i32,
    #[serde(default = "default_mamba_hidden_act")]
    mamba_hidden_act: String,
    #[serde(default = "default_time_step_min")]
    time_step_min: f32,
    #[serde(default = "default_time_step_max")]
    time_step_max: f32,
    #[serde(default = "default_time_step_floor")]
    time_step_floor: f32,
    #[serde(default = "default_true")]
    use_conv_bias: bool,
    #[serde(default)]
    mamba_proj_bias: bool,
    #[serde(default = "default_chunk_size")]
    chunk_size: i32,
    #[serde(default = "default_true")]
    rescale_prenorm_residual: bool,
    #[serde(default = "default_mlp_hidden_act")]
    mlp_hidden_act: String,
    #[serde(default = "default_n_routed_experts")]
    n_routed_experts: i32,
    #[serde(default = "default_n_shared_experts")]
    n_shared_experts: i32,
    #[serde(default = "default_moe_intermediate_size")]
    moe_intermediate_size: i32,
    #[serde(default = "default_moe_intermediate_size")]
    moe_shared_expert_intermediate_size: i32,
    #[serde(default = "default_num_experts_per_tok")]
    num_experts_per_tok: i32,
    #[serde(default = "default_routed_scaling_factor")]
    routed_scaling_factor: f32,
    #[serde(default = "default_n_group")]
    n_group: i32,
    #[serde(default = "default_topk_group")]
    topk_group: i32,
    #[serde(default = "default_true")]
    norm_topk_prob: bool,
    #[serde(default)]
    torch_dtype: Option<String>,
    #[serde(default)]
    quantization: Option<AffineQuantization>,
}

/// Validated Nemotron-H family policy.
#[derive(Debug, Clone)]
#[allow(missing_docs)]
pub struct ModelArgs {
    pub model_type: String,
    pub vocab_size: i32,
    pub tie_word_embeddings: bool,
    pub hidden_size: i32,
    pub intermediate_size: i32,
    pub num_hidden_layers: i32,
    pub num_nextn_predict_layers: i32,
    pub mtp_hybrid_override_pattern: Option<String>,
    pub layer_schedule: LayerSchedule<LayerPolicy>,
    pub num_attention_heads: i32,
    pub head_dim: i32,
    pub num_key_value_heads: i32,
    pub rope_theta: f32,
    pub max_position_embeddings: i32,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub use_bias: bool,
    pub layer_norm_epsilon: f32,
    pub norm_eps: f32,
    pub residual_in_fp32: bool,
    pub num_logits_to_keep: i32,
    pub ssm_state_size: i32,
    pub mamba_num_heads: i32,
    pub n_groups: i32,
    pub mamba_head_dim: i32,
    pub conv_kernel: i32,
    pub expand: i32,
    pub mamba_hidden_act: String,
    pub time_step_min: f32,
    pub time_step_max: f32,
    pub time_step_floor: f32,
    pub use_conv_bias: bool,
    pub mamba_proj_bias: bool,
    pub chunk_size: i32,
    pub rescale_prenorm_residual: bool,
    pub mlp_hidden_act: String,
    pub n_routed_experts: i32,
    pub n_shared_experts: i32,
    pub moe_intermediate_size: i32,
    pub moe_shared_expert_intermediate_size: i32,
    pub num_experts_per_tok: i32,
    pub routed_scaling_factor: f32,
    pub n_group: i32,
    pub topk_group: i32,
    pub norm_topk_prob: bool,
    pub weight_dtype: WeightDtype,
    pub weight_quantization: Option<WeightQuantization>,
    pub quantized_weights: Option<HashSet<String>>,
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

impl ModelArgs {
    /// Validates the normalized target, MTP, recurrent, attention, and expert geometry.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.model_type != "nemotron_h" {
            return Err(invalid(format!(
                "unsupported model_type {:?}",
                self.model_type
            )));
        }
        for (name, value) in [
            ("vocab_size", self.vocab_size),
            ("hidden_size", self.hidden_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("head_dim", self.head_dim),
            ("ssm_state_size", self.ssm_state_size),
            ("mamba_num_heads", self.mamba_num_heads),
            ("n_groups", self.n_groups),
            ("mamba_head_dim", self.mamba_head_dim),
            ("conv_kernel", self.conv_kernel),
            ("chunk_size", self.chunk_size),
            ("n_routed_experts", self.n_routed_experts),
            ("n_shared_experts", self.n_shared_experts),
            ("num_experts_per_tok", self.num_experts_per_tok),
        ] {
            if value <= 0 {
                return Err(invalid(format!("{name} must be positive, got {value}")));
            }
        }
        if self.layer_schedule.len() != self.num_hidden_layers as usize
            || self.num_attention_heads % self.num_key_value_heads != 0
            || self.mamba_num_heads % self.n_groups != 0
            || self.num_experts_per_tok > self.n_routed_experts
            || self.n_routed_experts % self.n_group != 0
            || self.topk_group > self.n_group
            || self.num_experts_per_tok > self.topk_group * (self.n_routed_experts / self.n_group)
        {
            return Err(invalid("invalid Nemotron-H schedule/head/expert geometry"));
        }
        if self.n_shared_experts != 1
            || self.mlp_hidden_act != "relu2"
            || self.mamba_hidden_act != "silu"
        {
            return Err(invalid(
                "Nemotron-H requires one shared expert, relu2 MLP, and silu Mamba",
            ));
        }
        if self.num_nextn_predict_layers < 0 {
            return Err(invalid("num_nextn_predict_layers cannot be negative"));
        }
        self.mtp_policies()?;
        Ok(())
    }

    /// Expands the repeated MTP operator pattern into physical unit policies.
    pub fn mtp_policies(&self) -> Result<Vec<LayerPolicy>, ConfigError> {
        if self.num_nextn_predict_layers == 0 {
            return Ok(Vec::new());
        }
        let pattern = self
            .mtp_hybrid_override_pattern
            .as_deref()
            .filter(|pattern| !pattern.is_empty())
            .ok_or_else(|| invalid("MTP weights require a nonempty MTP operator pattern"))?;
        let attention = self
            .layer_schedule
            .iter()
            .find_map(|policy| match policy {
                LayerPolicy::SelfAttention(policy) => Some(*policy),
                _ => None,
            })
            .unwrap_or(AttentionPolicy::Full);
        let mut policies = Vec::new();
        for _ in 0..self.num_nextn_predict_layers {
            for marker in pattern.chars() {
                let policy = LayerPolicy::from_marker(marker, attention)?;
                if !matches!(
                    policy,
                    LayerPolicy::SelfAttention(_) | LayerPolicy::SparseMoe
                ) {
                    return Err(invalid("MTP pattern supports only attention and MoE units"));
                }
                policies.push(policy);
            }
        }
        Ok(policies)
    }

    /// Returns whether target or prediction groups contain sparse experts.
    pub fn has_sparse_moe_layers(&self) -> bool {
        self.layer_schedule
            .iter()
            .any(|policy| *policy == LayerPolicy::SparseMoe)
            || self
                .mtp_policies()
                .is_ok_and(|policies| policies.contains(&LayerPolicy::SparseMoe))
    }

    /// Resolves the physical encoding for one canonical parameter identity.
    pub fn weight_quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        if let Some(configs) = &self.quantized_weight_configs {
            return configs.get(name).copied();
        }
        let quantization = self.weight_quantization?;
        match &self.quantized_weights {
            Some(names) if !names.contains(name) => None,
            _ => Some(quantization),
        }
    }
}

impl Source {
    fn normalize(self) -> Result<ModelArgs, ConfigError> {
        let attention = match self.sliding_window {
            None => AttentionPolicy::Full,
            Some(window) => AttentionPolicy::Sliding {
                window: NonZeroU32::new(u32::try_from(window).map_err(|_| {
                    invalid(format!("sliding_window must be positive, got {window}"))
                })?)
                .ok_or_else(|| invalid("sliding_window must be positive"))?,
            },
        };
        let count = usize::try_from(self.num_hidden_layers)
            .map_err(|_| invalid("num_hidden_layers must be positive"))?;
        let policies = self
            .hybrid_override_pattern
            .chars()
            .map(|marker| LayerPolicy::from_marker(marker, attention))
            .collect::<Result<Vec<_>, _>>()?;
        let layer_schedule = LayerSchedule::new(count, policies)
            .map_err(|error| invalid(format!("hybrid_override_pattern {error}")))?;
        let mtp_hybrid_override_pattern =
            match (self.mtp_hybrid_override_pattern, self.mtp_layers_block_type) {
                (Some(pattern), _) => Some(pattern),
                (None, Some(layers)) => Some(
                    layers
                        .iter()
                        .map(|layer| match layer.as_str() {
                            "attention" | "full_attention" => Ok('*'),
                            "moe" => Ok('E'),
                            other => Err(invalid(format!("unsupported MTP block type {other:?}"))),
                        })
                        .collect::<Result<String, _>>()?,
                ),
                (None, None) => None,
            };
        let weight_dtype = match self.torch_dtype.as_deref() {
            None | Some("float32") => WeightDtype::Float32,
            Some("float16") => WeightDtype::Float16,
            Some("bfloat16" | "bf16") => WeightDtype::Bfloat16,
            Some(dtype) => return Err(invalid(format!("unsupported torch_dtype {dtype:?}"))),
        };
        let args = ModelArgs {
            model_type: self.model_type,
            vocab_size: self.vocab_size,
            tie_word_embeddings: self.tie_word_embeddings,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_nextn_predict_layers: self.num_nextn_predict_layers,
            mtp_hybrid_override_pattern,
            layer_schedule,
            num_attention_heads: self.num_attention_heads,
            head_dim: self.head_dim,
            num_key_value_heads: self.num_key_value_heads,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            attention_bias: self.attention_bias,
            mlp_bias: self.mlp_bias,
            use_bias: self.use_bias,
            layer_norm_epsilon: self.layer_norm_epsilon,
            norm_eps: self.norm_eps,
            residual_in_fp32: self.residual_in_fp32,
            num_logits_to_keep: self.num_logits_to_keep,
            ssm_state_size: self.ssm_state_size,
            mamba_num_heads: self.mamba_num_heads,
            n_groups: self.n_groups,
            mamba_head_dim: self.mamba_head_dim,
            conv_kernel: self.conv_kernel,
            expand: self.expand,
            mamba_hidden_act: self.mamba_hidden_act,
            time_step_min: self.time_step_min,
            time_step_max: self.time_step_max,
            time_step_floor: self.time_step_floor,
            use_conv_bias: self.use_conv_bias,
            mamba_proj_bias: self.mamba_proj_bias,
            chunk_size: self.chunk_size,
            rescale_prenorm_residual: self.rescale_prenorm_residual,
            mlp_hidden_act: self.mlp_hidden_act,
            n_routed_experts: self.n_routed_experts,
            n_shared_experts: self.n_shared_experts,
            moe_intermediate_size: self.moe_intermediate_size,
            moe_shared_expert_intermediate_size: self.moe_shared_expert_intermediate_size,
            num_experts_per_tok: self.num_experts_per_tok,
            routed_scaling_factor: self.routed_scaling_factor,
            n_group: self.n_group,
            topk_group: self.topk_group,
            norm_topk_prob: self.norm_topk_prob,
            weight_dtype,
            weight_quantization: self.quantization.map(Into::into),
            quantized_weights: None,
            quantized_weight_configs: None,
        };
        args.validate()?;
        Ok(args)
    }
}

/// Parses and normalizes a Hugging Face configuration stream.
pub fn model_args_from_config_reader(reader: impl Read) -> Result<ModelArgs, ConfigError> {
    serde_json::from_reader::<_, Source>(reader)
        .map_err(|error| ConfigError::Decode(error.to_string()))?
        .normalize()
}

/// Parses and normalizes a Hugging Face configuration value.
pub fn model_args_from_config_value(value: &Value) -> Result<ModelArgs, ConfigError> {
    serde_json::from_value::<Source>(value.clone())
        .map_err(|error| ConfigError::Decode(error.to_string()))?
        .normalize()
}
/// Minimal physical tensor catalog required by GGUF normalization.
pub trait GgufTensorCatalog {
    /// Whether one exact physical tensor is present.
    fn contains(&self, name: &str) -> bool;
    /// Whether any physical tensor name satisfies the predicate.
    fn any(&self, predicate: impl FnMut(&str) -> bool) -> bool;
}

/// Normalizes model arguments from pure GGUF metadata and tensor presence.
pub fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "nemotron_h" | "nemotron_h_moe") {
        return Err(invalid(format!(
            "GGUF architecture {architecture:?}; this loader supports nemotron_h and nemotron_h_moe"
        )));
    }
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let is_moe = architecture == "nemotron_h_moe";
    let expert_count_key = key("expert_count");
    let has_experts = gguf_optional_i64(metadata, &expert_count_key)?.unwrap_or(0) > 0
        || arrays.any(|name| name.contains("_exps"));
    if is_moe != has_experts {
        return Err(invalid(
            "Nemotron-H GGUF architecture and expert tensors disagree",
        ));
    }
    let latent_size_key = key("moe_latent_size");
    if gguf_optional_i64(metadata, &latent_size_key)?.unwrap_or(0) > 0
        || arrays.any(|name| name.contains("ffn_latent_"))
    {
        return Err(invalid(
            "Nemotron-H latent-space MoE GGUF checkpoints are not supported",
        ));
    }

    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    ensure_positive("block_count", num_hidden_layers)?;
    let feed_forward_lengths = expand_layer_values(
        &key("feed_forward_length"),
        gguf_i64_values(metadata, &key("feed_forward_length"))?,
        num_hidden_layers,
    )?;
    let kv_head_counts = expand_layer_values(
        &key("attention.head_count_kv"),
        gguf_i64_values(metadata, &key("attention.head_count_kv"))?,
        num_hidden_layers,
    )?;
    let hybrid_override_pattern =
        hybrid_pattern_from_gguf_layers(&feed_forward_lengths, &kv_head_counts, is_moe);
    let intermediate_size =
        unique_nonzero_layer_value(&key("feed_forward_length"), &feed_forward_lengths)?;
    let num_key_value_heads =
        unique_nonzero_layer_value(&key("attention.head_count_kv"), &kv_head_counts)?;

    let inner_size = gguf_i32(metadata, &key("ssm.inner_size"))?;
    let mamba_num_heads = gguf_i32(metadata, &key("ssm.time_step_rank"))?;
    ensure_positive("ssm.inner_size", inner_size)?;
    ensure_positive("ssm.time_step_rank", mamba_num_heads)?;
    if inner_size % mamba_num_heads != 0 {
        return Err(invalid(format!(
            "Nemotron-H SSM inner size {inner_size} is not divisible by {mamba_num_heads} heads"
        )));
    }
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
    ensure_positive("embedding_length", hidden_size)?;
    ensure_positive("attention.head_count", num_attention_heads)?;
    let head_dim = gguf_optional_i64(metadata, &key("attention.key_length"))?
        .map(i32::try_from)
        .transpose()
        .map_err(|_| invalid("Nemotron-H head size exceeds i32"))?
        .unwrap_or(hidden_size / num_attention_heads);
    let norm_eps = gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?;
    let vocab_size = match metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
    {
        Some(tokens) => i32::try_from(tokens.len())
            .map_err(|_| invalid("GGUF tokenizer vocabulary exceeds i32"))?,
        None if metadata.contains_key("tokenizer.ggml.tokens") => {
            return Err(invalid(
                "GGUF tokenizer.ggml.tokens metadata has the wrong type",
            ));
        }
        None => gguf_i32(metadata, &key("vocab_size"))?,
    };

    let n_routed_experts = if is_moe {
        gguf_i32(metadata, &key("expert_count"))?
    } else {
        default_n_routed_experts()
    };
    let n_shared_experts = if is_moe {
        gguf_optional_i64(metadata, &key("expert_shared_count"))?
            .unwrap_or(1)
            .try_into()
            .map_err(|_| invalid("expert_shared_count exceeds i32"))?
    } else {
        default_n_shared_experts()
    };
    let moe_intermediate_size = if is_moe {
        gguf_i32(metadata, &key("expert_feed_forward_length"))?
    } else {
        default_moe_intermediate_size()
    };
    let moe_shared_expert_intermediate_size = if is_moe {
        gguf_i32(metadata, &key("expert_shared_feed_forward_length"))?
    } else {
        default_moe_intermediate_size()
    };
    let num_experts_per_tok = if is_moe {
        gguf_i32(metadata, &key("expert_used_count"))?
    } else {
        default_num_experts_per_tok()
    };

    let args = Source {
        model_type: "nemotron_h".into(),
        vocab_size,
        tie_word_embeddings: !arrays.contains("output.weight"),
        hidden_size,
        intermediate_size,
        num_hidden_layers,
        num_nextn_predict_layers: 0,
        mtp_hybrid_override_pattern: None,
        mtp_layers_block_type: None,
        hybrid_override_pattern,
        num_attention_heads,
        head_dim,
        num_key_value_heads,
        rope_theta: gguf_optional_f32(metadata, &key("rope.freq_base"))?
            .unwrap_or_else(default_rope_theta),
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        attention_bias: arrays.any(|name| {
            name.ends_with("attn_q.bias")
                || name.ends_with("attn_k.bias")
                || name.ends_with("attn_v.bias")
                || name.ends_with("attn_output.bias")
        }),
        mlp_bias: arrays
            .any(|name| name.ends_with("ffn_up.bias") || name.ends_with("ffn_down.bias")),
        use_bias: arrays
            .any(|name| name.ends_with("ssm_in.bias") || name.ends_with("ssm_out.bias")),
        layer_norm_epsilon: norm_eps,
        norm_eps,
        residual_in_fp32: false,
        num_logits_to_keep: 1,
        sliding_window: gguf_optional_i64(metadata, &key("attention.sliding_window"))?,
        ssm_state_size: gguf_i32(metadata, &key("ssm.state_size"))?,
        mamba_num_heads,
        n_groups: gguf_i32(metadata, &key("ssm.group_count"))?,
        mamba_head_dim: inner_size / mamba_num_heads,
        conv_kernel: gguf_i32(metadata, &key("ssm.conv_kernel"))?,
        expand: 2,
        mamba_hidden_act: "silu".into(),
        time_step_min: default_time_step_min(),
        time_step_max: default_time_step_max(),
        time_step_floor: default_time_step_floor(),
        use_conv_bias: arrays.any(|name| name.ends_with("ssm_conv1d.bias")),
        mamba_proj_bias: arrays
            .any(|name| name.ends_with("ssm_in.bias") || name.ends_with("ssm_out.bias")),
        chunk_size: default_chunk_size(),
        rescale_prenorm_residual: true,
        mlp_hidden_act: "relu2".into(),
        n_routed_experts,
        n_shared_experts,
        moe_intermediate_size,
        moe_shared_expert_intermediate_size,
        num_experts_per_tok,
        routed_scaling_factor: if is_moe {
            gguf_optional_f32(metadata, &key("expert_weights_scale"))?.unwrap_or(1.0)
        } else {
            default_routed_scaling_factor()
        },
        n_group: if is_moe {
            gguf_optional_i64(metadata, &key("expert_group_count"))?
                .unwrap_or(1)
                .try_into()
                .map_err(|_| invalid("expert_group_count exceeds i32"))?
        } else {
            default_n_group()
        },
        topk_group: if is_moe {
            gguf_optional_i64(metadata, &key("expert_group_used_count"))?
                .unwrap_or(1)
                .try_into()
                .map_err(|_| invalid("expert_group_used_count exceeds i32"))?
        } else {
            default_topk_group()
        },
        norm_topk_prob: if is_moe {
            gguf_optional_i64(metadata, &key("expert_weights_norm"))?.unwrap_or(1) != 0
        } else {
            true
        },
        torch_dtype: None,
        quantization: None,
    }
    .normalize()?;
    args.validate()?;
    Ok(args)
}

fn expand_layer_values(
    key: &str,
    values: Vec<i64>,
    num_hidden_layers: i32,
) -> Result<Vec<i32>, ConfigError> {
    let values = if values.len() == 1 {
        vec![values[0]; num_hidden_layers as usize]
    } else if values.len() == num_hidden_layers as usize {
        values
    } else {
        return Err(invalid(format!(
            "GGUF metadata key {key:?} has {} values for {num_hidden_layers} layers",
            values.len()
        )));
    };
    values
        .into_iter()
        .map(|value| {
            i32::try_from(value)
                .map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
        })
        .collect()
}

fn hybrid_pattern_from_gguf_layers(
    feed_forward_lengths: &[i32],
    kv_head_counts: &[i32],
    is_moe: bool,
) -> String {
    feed_forward_lengths
        .iter()
        .zip(kv_head_counts)
        .map(|(feed_forward, kv_heads)| {
            if *feed_forward > 0 {
                if is_moe {
                    'E'
                } else {
                    '-'
                }
            } else if *kv_heads > 0 {
                '*'
            } else {
                'M'
            }
        })
        .collect()
}

fn unique_nonzero_layer_value(key: &str, values: &[i32]) -> Result<i32, ConfigError> {
    let Some(value) = values.iter().copied().find(|value| *value > 0) else {
        return Err(invalid(format!(
            "GGUF metadata key {key:?} has no non-zero layer value"
        )));
    };
    if values.iter().any(|other| *other > 0 && *other != value) {
        return Err(invalid(format!(
            "GGUF metadata key {key:?} has non-uniform non-zero layer values"
        )));
    }
    Ok(value)
}
fn gguf_string(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<String, ConfigError> {
    match metadata.get(key) {
        Some(MetadataValue::String(value)) => Ok(value.clone()),
        Some(_) => Err(invalid(format!(
            "GGUF metadata key {key:?} has the wrong type"
        ))),
        None => Err(invalid(format!(
            "GGUF metadata is missing required key {key:?}"
        ))),
    }
}

fn gguf_i32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i32, ConfigError> {
    i32::try_from(gguf_i64(metadata, key)?)
        .map_err(|_| invalid(format!("GGUF metadata value {key:?} exceeds i32")))
}

fn gguf_i64(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<i64, ConfigError> {
    gguf_optional_i64(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_i64(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<i64>, ConfigError> {
    let Some(values) = gguf_optional_i64_values(metadata, key)? else {
        return Ok(None);
    };
    if values.len() != 1 {
        return Err(invalid(format!("GGUF metadata key {key:?} must be scalar")));
    }
    Ok(values.into_iter().next())
}

fn gguf_i64_values(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Vec<i64>, ConfigError> {
    gguf_optional_i64_values(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_i64_values(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<Vec<i64>>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value
            .to_i64_vec()
            .map(Some)
            .ok_or_else(|| invalid(format!("GGUF metadata key {key:?} has the wrong type"))),
        None => Ok(None),
    }
}

fn gguf_f32(metadata: &HashMap<String, MetadataValue>, key: &str) -> Result<f32, ConfigError> {
    gguf_optional_f32(metadata, key)?
        .ok_or_else(|| invalid(format!("GGUF metadata is missing required key {key:?}")))
}

fn gguf_optional_f32(
    metadata: &HashMap<String, MetadataValue>,
    key: &str,
) -> Result<Option<f32>, ConfigError> {
    match metadata.get(key) {
        Some(value) => value.as_f32().map(Some).ok_or_else(|| {
            invalid(format!(
                "GGUF metadata key {key:?} must be a numeric scalar"
            ))
        }),
        None => Ok(None),
    }
}

/// Declares global state for every target and appended prediction unit.
pub fn state_layout(args: &ModelArgs) -> Result<StateLayout, ConfigError> {
    let mut geometry = args
        .layer_schedule
        .iter()
        .map(|policy| match policy {
            LayerPolicy::Mamba => LayerGeometry::Mamba {
                heads: args.mamba_num_heads,
                groups: args.n_groups,
            },
            LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
                query_heads: args.num_attention_heads,
                kv_heads: args.num_key_value_heads,
            },
            LayerPolicy::DenseMlp => LayerGeometry::DenseMlp {
                intermediate: args.intermediate_size,
            },
            LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
                routed: args.moe_intermediate_size,
                shared: args.moe_shared_expert_intermediate_size,
            },
        })
        .collect::<Vec<_>>();
    geometry.extend(args.mtp_policies()?.into_iter().map(|policy| match policy {
        LayerPolicy::SelfAttention(_) => LayerGeometry::Attention {
            query_heads: args.num_attention_heads,
            kv_heads: args.num_key_value_heads,
        },
        LayerPolicy::SparseMoe => LayerGeometry::SparseMoe {
            routed: args.moe_intermediate_size,
            shared: args.moe_shared_expert_intermediate_size,
        },
        _ => unreachable!("validated MTP policies contain only attention and MoE"),
    }));
    state_layout_with_geometry(args, &geometry)
}

/// Declares rank-local state from placement-resolved unit geometry.
pub fn state_layout_with_geometry(
    args: &ModelArgs,
    geometry: &[LayerGeometry],
) -> Result<StateLayout, ConfigError> {
    let mut schedule = args.layer_schedule.iter().copied().collect::<Vec<_>>();
    schedule.extend(args.mtp_policies()?);
    if geometry.len() != schedule.len() {
        return Err(invalid(
            "state geometry does not match the target plus MTP physical schedule",
        ));
    }
    let history = args.conv_kernel - 1;
    let fixed = |value| StateTensorDimension::fixed(value).map_err(|e| invalid(e.to_string()));
    let policies = schedule
        .iter()
        .zip(geometry)
        .map(|(policy, geometry)| match (*policy, *geometry) {
            (LayerPolicy::Mamba, LayerGeometry::Mamba { heads, groups }) => {
                let conv_width = heads * args.mamba_head_dim + 2 * groups * args.ssm_state_size;
                let mut tensors = Vec::new();
                if history > 0 {
                    tensors.push(
                        StateTensorPolicy::new(
                            StateTensorRole::Convolution { slot: 0 },
                            vec![
                                StateTensorDimension::Batch,
                                fixed(history)?,
                                fixed(conv_width)?,
                            ],
                            StateTensorDtype::Floating,
                            MutableStateResidency::AlwaysDeviceMutable,
                        )
                        .map_err(|e| invalid(e.to_string()))?,
                    );
                }
                tensors.push(
                    StateTensorPolicy::new(
                        StateTensorRole::Recurrent,
                        vec![
                            StateTensorDimension::Batch,
                            fixed(heads)?,
                            fixed(args.mamba_head_dim)?,
                            fixed(args.ssm_state_size)?,
                        ],
                        StateTensorDtype::Float32,
                        MutableStateResidency::LayerScopedOffloadable,
                    )
                    .map_err(|e| invalid(e.to_string()))?,
                );
                LayerCachePolicy::fixed_only(tensors).map_err(|e| invalid(e.to_string()))
            }
            (LayerPolicy::SelfAttention(attention), LayerGeometry::Attention { kv_heads, .. }) => {
                LayerCachePolicy::key_value(attention, kv_heads, args.head_dim)
                    .map_err(|e| invalid(e.to_string()))
            }
            (LayerPolicy::DenseMlp, LayerGeometry::DenseMlp { .. })
            | (LayerPolicy::SparseMoe, LayerGeometry::SparseMoe { .. }) => {
                Ok(LayerCachePolicy::NoState)
            }
            _ => Err(invalid(
                "state geometry does not match its scheduled operator",
            )),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let target_layers = args.layer_schedule.len();
    let prediction_layers = policies.len() - target_layers;
    let mut segments = vec![StateSegmentSpec::new(
        TARGET_STATE_SEGMENT,
        0..target_layers,
        StateSegmentLifetime::Persistent,
        0,
    )
    .map_err(|error| invalid(error.to_string()))?];
    if prediction_layers > 0 {
        segments.push(
            StateSegmentSpec::new(
                PREDICTION_STATE_SEGMENT,
                target_layers..target_layers + prediction_layers,
                StateSegmentLifetime::Persistent,
                -1,
            )
            .map_err(|error| invalid(error.to_string()))?,
        );
    }
    StateLayout::segmented(
        LayerSchedule::new(policies.len(), policies).map_err(|e| invalid(e.to_string()))?,
        segments,
    )
    .map_err(|e| invalid(e.to_string()))
}

/// Returns the stable prompt-cache fingerprint of state-affecting policy.
pub fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    derive_prompt_cache_architecture_fingerprint(
        "nemotron_h",
        [
            ("model_type", args.model_type.clone()),
            ("hidden", args.hidden_size.to_string()),
            (
                "schedule",
                args.layer_schedule
                    .iter()
                    .map(|p| match p {
                        LayerPolicy::Mamba => "m".into(),
                        LayerPolicy::SelfAttention(AttentionPolicy::Full) => "af".into(),
                        LayerPolicy::SelfAttention(AttentionPolicy::Sliding { window }) => {
                            format!("as{}", window.get())
                        }
                        LayerPolicy::DenseMlp => "d".into(),
                        LayerPolicy::SparseMoe => "e".into(),
                    })
                    .collect::<Vec<_>>()
                    .join(","),
            ),
            ("mamba_state", args.ssm_state_size.to_string()),
            ("mamba_heads", args.mamba_num_heads.to_string()),
            ("mamba_groups", args.n_groups.to_string()),
            ("mamba_head_dim", args.mamba_head_dim.to_string()),
            ("mamba_conv", args.conv_kernel.to_string()),
            ("residual_f32", args.residual_in_fp32.to_string()),
            (
                "mtp",
                args.mtp_hybrid_override_pattern.clone().unwrap_or_default(),
            ),
        ],
    )
}

/// Invalid or unsupported Nemotron-H configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// JSON decoding failed before semantic normalization.
    #[error("invalid Nemotron-H JSON: {0}")]
    Decode(String),
    /// The decoded or GGUF-derived policy is unsupported or inconsistent.
    #[error("unsupported Nemotron-H configuration: {0}")]
    Invalid(String),
}

fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError::Invalid(message.into())
}
fn ensure_positive(name: &str, value: i32) -> Result<(), ConfigError> {
    if value <= 0 {
        Err(invalid(format!("{name} must be positive, got {value}")))
    } else {
        Ok(())
    }
}
fn default_true() -> bool {
    true
}
fn default_vocab_size() -> i32 {
    131_072
}
fn default_hidden_size() -> i32 {
    4096
}
fn default_intermediate_size() -> i32 {
    21_504
}
fn default_num_hidden_layers() -> i32 {
    52
}
fn default_hybrid_override_pattern() -> String {
    "M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M*-M-M-M-M-M-".into()
}
fn default_num_attention_heads() -> i32 {
    32
}
fn default_head_dim() -> i32 {
    128
}
fn default_num_key_value_heads() -> i32 {
    8
}
fn default_rope_theta() -> f32 {
    10_000.0
}
fn default_max_position_embeddings() -> i32 {
    4096
}
fn default_norm_eps() -> f32 {
    1e-5
}
fn default_num_logits_to_keep() -> i32 {
    1
}
fn default_ssm_state_size() -> i32 {
    128
}
fn default_mamba_num_heads() -> i32 {
    128
}
fn default_n_groups() -> i32 {
    8
}
fn default_mamba_head_dim() -> i32 {
    64
}
fn default_conv_kernel() -> i32 {
    4
}
fn default_expand() -> i32 {
    2
}
fn default_mamba_hidden_act() -> String {
    "silu".into()
}
fn default_time_step_min() -> f32 {
    0.001
}
fn default_time_step_max() -> f32 {
    0.1
}
fn default_time_step_floor() -> f32 {
    0.0001
}
fn default_chunk_size() -> i32 {
    128
}
fn default_mlp_hidden_act() -> String {
    "relu2".into()
}
fn default_n_routed_experts() -> i32 {
    8
}
fn default_n_shared_experts() -> i32 {
    1
}
fn default_moe_intermediate_size() -> i32 {
    7688
}
fn default_num_experts_per_tok() -> i32 {
    2
}
fn default_routed_scaling_factor() -> f32 {
    1.0
}
fn default_n_group() -> i32 {
    1
}
fn default_topk_group() -> i32 {
    1
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_physical_and_mtp_patterns() {
        let args = model_args_from_config_value(&serde_json::json!({
            "model_type":"nemotron_h", "vocab_size":32, "hidden_size":16,
            "intermediate_size":24, "num_hidden_layers":4,
            "hybrid_override_pattern":"M*-E", "num_attention_heads":4,
            "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
            "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
            "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
            "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
            "num_experts_per_tok":2, "n_group":2, "topk_group":1,
            "num_nextn_predict_layers":1, "mtp_hybrid_override_pattern":"*E"
        }))
        .unwrap();
        assert_eq!(args.layer_schedule.len(), 4);
        assert_eq!(args.mtp_policies().unwrap().len(), 2);
        assert!(matches!(
            state_layout(&args).unwrap().layers().get(0),
            Some(LayerCachePolicy::FixedState { .. })
        ));
        let layout = state_layout(&args).unwrap();
        assert_eq!(layout.segments().len(), 2);
        assert_eq!(layout.segments()[0].id().as_str(), TARGET_STATE_SEGMENT);
        assert_eq!(layout.segments()[0].layers(), 0..4);
        assert_eq!(layout.segments()[1].id().as_str(), PREDICTION_STATE_SEGMENT);
        assert_eq!(layout.segments()[1].layers(), 4..6);
        let identity = crate::nemotron_h::state_identity(
            &args,
            &layout,
            0,
            eredu_core::cache::PromptCacheTopology::default(),
        )
        .unwrap()
        .prompt_cache_identity(&layout)
        .unwrap();
        assert_eq!(identity.layer_prefix_offsets, [0, 0, 0, 0, -1, -1]);
    }
}
