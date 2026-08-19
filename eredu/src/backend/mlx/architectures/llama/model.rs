//! Llama decoder-only model implementation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::{Module, ModuleParametersExt},
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use crate::backend::mlx::nn::generation::sample;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::tensor::{
        create_attention_mask,
        rope::{initialize_rope, validate_rope_scaling_config, FloatOrString, RopeVariant},
        AttentionMask,
    },
    backend::mlx::nn::{
        self as common,
        attention::{
            apply_rope_and_update_cache, attention_probabilities, batch_seq, finish_attention,
            reshape_attention_projection,
        },
        generation::CausalLm,
        layers::SwiGluMlp,
        linear::project_logits_maybe_quantized,
    },
    backend::mlx::runtime::cache::{ConcatKeyValueCache, KeyValueCache},
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    backend::mlx::runtime::checkpoint::quantization::WeightQuantization,
    backend::mlx::runtime::execution::inspection::ActivationObserver,
    backend::mlx::runtime::media::input,
    core::attention::{AttentionPolicy, LayerSchedule},
    core::cache::derive_prompt_cache_architecture_fingerprint,
};
#[derive(Debug, Clone)]
/// Normalized Llama/Mistral decoder geometry used by every execution path.
pub struct ModelArgs {
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
    /// Whether RoPE uses adjacent-pair ordering instead of split-half ordering.
    pub rope_traditional: bool,
    /// Per-head attention dimension.
    pub head_dim: i32,
    /// Whether logits use tied input embeddings.
    pub tie_word_embeddings: bool,
    /// Whether attention projection layers include bias terms.
    pub attention_bias: bool,
    /// Whether MLP projection layers include bias terms.
    pub mlp_bias: bool,
    /// Optional RoPE scaling configuration.
    pub rope_scaling: Option<HashMap<String, FloatOrString>>,
    /// Exact ordered attention behavior for every decoder layer.
    pub attention_schedule: LayerSchedule<AttentionPolicy>,
    /// Preferred MLX-LM affine quantization metadata.
    pub quantization: Option<WeightQuantization>,
    /// Hugging Face-compatible alias emitted by MLX-LM converters.
    pub quantization_config: Option<WeightQuantization>,
    /// Optional exact weight names that use affine quantization.
    ///
    /// `None` preserves MLX-LM's model-wide quantization behavior. GGUF
    /// loading uses `Some` to represent files containing a mixture of packed
    /// and dense matrices.
    pub quantized_weights: Option<HashSet<String>>,
    /// Exact affine settings for mixed GGUF tensors.
    pub quantized_weight_configs: Option<HashMap<String, WeightQuantization>>,
}

#[derive(Debug, Clone, Deserialize)]
struct ModelArgsSource {
    model_type: String,
    hidden_size: i32,
    num_hidden_layers: i32,
    intermediate_size: i32,
    num_attention_heads: i32,
    rms_norm_eps: f32,
    vocab_size: i32,
    #[serde(default)]
    num_key_value_heads: i32,
    #[serde(default)]
    max_position_embeddings: i32,
    #[serde(default = "default_rope_theta")]
    rope_theta: f32,
    #[serde(default)]
    rope_traditional: bool,
    #[serde(default)]
    head_dim: i32,
    #[serde(default = "default_true")]
    tie_word_embeddings: bool,
    #[serde(default)]
    attention_bias: bool,
    #[serde(default)]
    mlp_bias: bool,
    rope_scaling: Option<HashMap<String, FloatOrString>>,
    #[serde(default)]
    sliding_window: Option<Value>,
    #[serde(default)]
    quantization: Option<WeightQuantization>,
    #[serde(default)]
    quantization_config: Option<WeightQuantization>,
}

impl ModelArgs {
    pub(crate) fn weight_quantization(&self) -> Option<WeightQuantization> {
        self.quantization.or(self.quantization_config)
    }

    pub(crate) fn affine_quantization_for(&self, weight_name: &str) -> Option<WeightQuantization> {
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
}

pub(crate) fn prompt_cache_architecture_fingerprint(args: &ModelArgs) -> String {
    let rope_scaling = args.rope_scaling.as_ref().map_or_else(
        || "none".to_string(),
        |config| {
            let mut entries = config.iter().collect::<Vec<_>>();
            entries.sort_unstable_by_key(|(key, _)| key.as_str());
            entries
                .into_iter()
                .map(|(key, value)| {
                    let value = match value {
                        FloatOrString::Float(value) => format!("f32:{:08x}", value.to_bits()),
                        FloatOrString::String(value) => format!("string:{value}"),
                        FloatOrString::Bool(value) => format!("bool:{value}"),
                    };
                    format!("{key}={value}")
                })
                .collect::<Vec<_>>()
                .join(";")
        },
    );
    let mut quantized_weights = args
        .quantized_weights
        .as_ref()
        .map(|weights| weights.iter().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    quantized_weights.sort_unstable();
    let mut quantized_weight_configs = args
        .quantized_weight_configs
        .as_ref()
        .map(|configs| {
            configs
                .iter()
                .map(|(name, config)| format!("{name}={config:?}"))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    quantized_weight_configs.sort_unstable();
    derive_prompt_cache_architecture_fingerprint(
        "llama",
        [
            ("model_type", args.model_type.clone()),
            ("hidden_size", args.hidden_size.to_string()),
            ("num_hidden_layers", args.num_hidden_layers.to_string()),
            ("intermediate_size", args.intermediate_size.to_string()),
            ("num_attention_heads", args.num_attention_heads.to_string()),
            ("num_key_value_heads", args.num_key_value_heads.to_string()),
            ("head_dim", args.head_dim.to_string()),
            (
                "rms_norm_eps",
                format!("{:08x}", args.rms_norm_eps.to_bits()),
            ),
            ("vocab_size", args.vocab_size.to_string()),
            (
                "max_position_embeddings",
                args.max_position_embeddings.to_string(),
            ),
            ("rope_theta", format!("{:08x}", args.rope_theta.to_bits())),
            ("rope_traditional", args.rope_traditional.to_string()),
            ("rope_scaling", rope_scaling),
            (
                "attention_schedule",
                args.attention_schedule.fingerprint_component(),
            ),
            ("tie_word_embeddings", args.tie_word_embeddings.to_string()),
            ("attention_bias", args.attention_bias.to_string()),
            ("mlp_bias", args.mlp_bias.to_string()),
            ("quantization", format!("{:?}", args.weight_quantization())),
            ("quantized_weights", quantized_weights.join(";")),
            (
                "quantized_weight_configs",
                quantized_weight_configs.join(";"),
            ),
        ],
    )
}

fn default_true() -> bool {
    true
}

fn default_rope_theta() -> f32 {
    10_000.0
}

/// Internal input shared by Llama-compatible attention and decoder blocks.
pub struct AttentionInput<'a, C> {
    /// Hidden states with shape `[batch, sequence, hidden]`.
    pub x: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Optional mutable key/value cache.
    pub cache: Option<&'a mut C>,
    /// Whether a layer may generate its canonical sliding-prefill mask.
    pub allow_sliding_prefill: bool,
}

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Llama attention layer.
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
    /// Rotary position embedding module.
    pub rope: RopeVariant,
    /// Layer-local attention window; absent for full causal attention.
    pub sliding_window: Option<i32>,
}

impl Attention {
    /// Creates an unloaded attention layer from model arguments.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let policy = args.attention_schedule.get(0).ok_or_else(|| {
            Exception::custom("Llama attention schedule has no policy for layer 0")
        })?;
        Self::new_with_prefix(args, None, *policy, stream)
    }

    fn new_for_layer(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer = usize::try_from(layer_index)
            .map_err(|_| Exception::custom(format!("invalid Llama layer index {layer_index}")))?;
        let policy = args.attention_schedule.get(layer).ok_or_else(|| {
            Exception::custom(format!(
                "Llama attention schedule has no policy for layer {layer_index}"
            ))
        })?;
        Self::new_with_prefix(
            args,
            Some(format!("model.layers.{layer_index}.self_attn")),
            *policy,
            stream,
        )
    }

    fn new_with_prefix(
        args: &ModelArgs,
        prefix: Option<String>,
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
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.q_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;
        let k_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.k_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;
        let v_proj = common::linear::unloaded_maybe_quantized_linear(
            dim,
            n_kv_heads * head_dim,
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.v_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;
        let o_proj = common::linear::unloaded_maybe_quantized_linear(
            n_heads * head_dim,
            dim,
            args.attention_bias,
            prefix
                .as_ref()
                .and_then(|prefix| args.affine_quantization_for(&format!("{prefix}.o_proj.weight")))
                .or_else(|| {
                    prefix
                        .is_none()
                        .then(|| args.weight_quantization())
                        .flatten()
                }),
            stream,
        )?;

        let rope = initialize_rope(
            head_dim,
            args.rope_theta,
            args.rope_traditional,
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
            rope,
            sliding_window: policy
                .window()
                .map(|window| i32::try_from(window.get()))
                .transpose()
                .map_err(|_| Exception::custom("Llama sliding window exceeds i32"))?,
        })
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
        let AttentionInput {
            x,
            mask,
            mut cache,
            allow_sliding_prefill,
        } = input;

        let (batch, seq_len) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.q_proj"), &queries)?;
        let keys = self.k_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.k_proj"), &keys)?;
        let values = self.v_proj.forward(x, stream)?;
        observer.observe(&format!("{prefix}.v_proj"), &values)?;

        let queries = reshape_attention_projection(queries, batch, seq_len, self.n_heads, stream)?;
        observer.observe(&format!("{prefix}.queries"), &queries)?;
        let keys = reshape_attention_projection(keys, batch, seq_len, self.n_kv_heads, stream)?;
        observer.observe(&format!("{prefix}.keys"), &keys)?;
        let values = reshape_attention_projection(values, batch, seq_len, self.n_kv_heads, stream)?;
        observer.observe(&format!("{prefix}.values"), &values)?;

        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        observer.observe(&format!("{prefix}.queries_rope"), &queries)?;
        observer.observe(&format!("{prefix}.keys_rope"), &keys)?;
        observer.observe(&format!("{prefix}.values_cache"), &values)?;
        let output = if let Some(window) = self
            .sliding_window
            .filter(|_| allow_sliding_prefill && seq_len > 1)
        {
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
        let AttentionInput {
            x,
            mask,
            mut cache,
            allow_sliding_prefill,
        } = input;

        let (B, L) = batch_seq(x);

        let queries = self.q_proj.forward(x, stream)?;
        let keys = self.k_proj.forward(x, stream)?;
        let values = self.v_proj.forward(x, stream)?;

        let queries = reshape_attention_projection(queries, B, L, self.n_heads, stream)?;
        let keys = reshape_attention_projection(keys, B, L, self.n_kv_heads, stream)?;
        let values = reshape_attention_projection(values, B, L, self.n_kv_heads, stream)?;
        let position_offset = cache.as_ref().map_or(0, |cache| cache.offset());
        let (queries, keys, values) =
            apply_rope_and_update_cache(&mut self.rope, queries, keys, values, &mut cache, stream)?;
        let output = if let Some(window_size) = self
            .sliding_window
            .filter(|_| allow_sliding_prefill && L > 1)
        {
            common::attention::sliding_window_prefill_attention(
                queries,
                keys,
                values,
                self.scale,
                window_size,
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
        <RopeVariant as Module<nn::RopeInput>>::training_mode(&mut self.rope, mode);
    }
}

/// Llama feed-forward block.
pub type Mlp = SwiGluMlp;

#[derive(Debug, Clone, ModuleParameters, Quantizable)]
/// Llama decoder block.
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
    /// Feed-forward layer.
    pub mlp: Mlp,

    #[param]
    /// Pre-attention RMSNorm.
    pub input_layernorm: nn::RmsNorm,

    #[param]
    /// Pre-MLP RMSNorm.
    pub post_attention_layernorm: nn::RmsNorm,
}

impl TransformerBlock {
    /// Creates an unloaded decoder block from model arguments.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_for_layer(args, 0, stream)
    }

    pub(crate) fn new_for_layer(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let num_attention_heads = args.num_attention_heads;
        let hidden_size = args.hidden_size;

        let self_attn = Attention::new_for_layer(args, layer_index, stream)?;
        let mlp_prefix = format!("model.layers.{layer_index}.mlp");
        let mlp = SwiGluMlp {
            gate_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.intermediate_size,
                args.mlp_bias,
                args.affine_quantization_for(&format!("{mlp_prefix}.gate_proj.weight")),
                stream,
            )?,
            down_proj: common::linear::unloaded_maybe_quantized_linear(
                args.intermediate_size,
                args.hidden_size,
                args.mlp_bias,
                args.affine_quantization_for(&format!("{mlp_prefix}.down_proj.weight")),
                stream,
            )?,
            up_proj: common::linear::unloaded_maybe_quantized_linear(
                args.hidden_size,
                args.intermediate_size,
                args.mlp_bias,
                args.affine_quantization_for(&format!("{mlp_prefix}.up_proj.weight")),
                stream,
            )?,
        };
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
        let AttentionInput {
            x,
            mask,
            cache,
            allow_sliding_prefill,
        } = input;

        observer.observe(&format!("{prefix}.input"), x)?;
        observer.observe(&format!("{prefix}.residual_before_attention"), x)?;
        let normed = self.input_layernorm.forward(x, stream)?;
        observer.observe(&format!("{prefix}.input_layernorm"), &normed)?;

        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
            allow_sliding_prefill,
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

        observer.observe(&format!("{prefix}.residual_before_mlp"), &h)?;
        let post_normed = self.post_attention_layernorm.forward(&h, stream)?;
        observer.observe(&format!("{prefix}.post_attention_layernorm"), &post_normed)?;
        let r = self.mlp.forward_with_observer(
            &post_normed,
            stream,
            &format!("{prefix}.mlp"),
            observer,
        )?;
        observer.observe(&format!("{prefix}.mlp_output"), &r)?;
        observer.observe(&format!("{prefix}.residual_delta_mlp"), &r)?;
        let output = h.add(r, stream)?;
        let output = observer
            .intervene(&format!("{prefix}.output"), &output)?
            .unwrap_or(output);
        observer.observe(&format!("{prefix}.output"), &output)?;
        observer.observe(&format!("{prefix}.residual_after_mlp"), &output)?;
        Ok(output)
    }

    /// Executes a block whose attention heads and MLP intermediates are
    /// rank-local, reducing each row projection exactly once.
    pub(crate) fn forward_tensor_parallel<C: KeyValueCache>(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        allow_sliding_prefill: bool,
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
        let keys =
            reshape_attention_projection(keys, batch, sequence, attention.n_kv_heads, stream)?;
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
        let attended = if let Some(window) = attention
            .sliding_window
            .filter(|_| allow_sliding_prefill && sequence > 1)
        {
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
        let gate = crate::backend::mlx::nn::layers::silu(
            self.mlp.gate_proj.forward(&normalized, stream)?,
            stream,
        )?;
        let up = self.mlp.up_proj.forward(&normalized, stream)?;
        let mlp = crate::backend::mlx::nn::parallel::forward_row_parallel(
            &mut self.mlp.down_proj,
            &gate.multiply(up, stream)?,
            group,
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
        let AttentionInput {
            x,
            mask,
            cache,
            allow_sliding_prefill,
        } = input;

        let normed = self.input_layernorm.forward(x, stream)?;
        let self_attn_input = AttentionInput {
            x: &normed,
            mask,
            cache,
            allow_sliding_prefill,
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
/// Llama transformer body without the language-model head.
pub struct ResidentDecoder {
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

impl ResidentDecoder {
    /// Creates an unloaded Llama transformer body.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        assert!(args.vocab_size.is_positive());

        let vocab_size = args.vocab_size;
        let num_hidden_layers = args.num_hidden_layers;

        let embed_tokens = common::linear::unloaded_maybe_quantized_embedding(
            args.vocab_size,
            args.hidden_size,
            args.affine_quantization_for("model.embed_tokens.weight"),
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

    fn attention_mask<C>(
        &self,
        h: &Array,
        cache: &[Option<C>],
        stream: &Stream,
    ) -> Result<Option<Array>, Exception>
    where
        C: KeyValueCache,
    {
        match create_attention_mask(h, cache, Some(true), stream)? {
            Some(AttentionMask::Array(mask)) => Ok(Some(mask)),
            Some(AttentionMask::Causal) => Err(Exception::custom(
                "Llama-compatible decoders require an explicit attention mask",
            )),
            None => Ok(None),
        }
    }

    /// Forward pass that reports transformer-body activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        let ModelInput {
            inputs,
            mask,
            cache,
        } = input;

        let mut h = self.embed_tokens.forward(inputs, stream)?;
        observer.observe("model.embed_tokens", &h)?;

        let allow_sliding_prefill = mask.is_none();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => self.attention_mask(&h, cache, stream)?,
        };
        if let Some(mask) = mask.as_ref() {
            observer.observe("model.attention_mask", mask)?;
        }

        for (i, (layer, c)) in self.layers.iter_mut().zip(cache.iter_mut()).enumerate() {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
                allow_sliding_prefill,
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

/// Input for a Llama forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

impl<C> Module<ModelInput<'_, C>> for ResidentDecoder
where
    C: KeyValueCache,
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

        let allow_sliding_prefill = mask.is_none();
        let mask = match mask {
            Some(mask) => Some(mask.clone()),
            None => self.attention_mask(&h, cache, stream)?,
        };

        for (layer, c) in self.layers.iter_mut().zip(cache.iter_mut()) {
            let layer_input = AttentionInput {
                x: &h,
                mask: mask.as_ref(),
                cache: c.as_mut(),
                allow_sliding_prefill,
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
/// Llama causal language model.
pub struct ResidentModel {
    /// Model configuration.
    pub args: ModelArgs,

    #[quantizable]
    #[param]
    /// Transformer body.
    pub model: ResidentDecoder,

    #[quantizable]
    #[param]
    /// Optional untied language-model head.
    pub lm_head: Option<MaybeQuantized<nn::Linear>>,
}

impl ResidentModel {
    /// Creates an unloaded Llama causal language model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let model = ResidentDecoder::new(&args, stream)?;
        let lm_head = if !args.tie_word_embeddings {
            Some(
                common::linear::build_unloaded_maybe_quantized_lm_head_with_quantization(
                    args.hidden_size,
                    args.vocab_size,
                    args.affine_quantization_for("lm_head.weight"),
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

    /// Creates one architecture-correct device cache per decoder layer.
    pub fn new_cache(&self) -> Vec<Option<ConcatKeyValueCache>> {
        self.args
            .attention_schedule
            .iter()
            .map(|policy| {
                Some(match policy.window() {
                    Some(window) => ConcatKeyValueCache::new_for_sliding_attention(
                        i32::try_from(window.get())
                            .expect("validated Llama attention window fits i32"),
                    ),
                    None => ConcatKeyValueCache::new(),
                })
            })
            .collect()
    }

    fn validate_cache<C: KeyValueCache>(&self, cache: &[Option<C>]) -> Result<(), Exception> {
        if cache.len() != self.args.attention_schedule.len() {
            return Err(Exception::custom(format!(
                "Llama cache has {} layers, expected {}",
                cache.len(),
                self.args.attention_schedule.len()
            )));
        }
        for (layer, (cache, policy)) in cache
            .iter()
            .zip(self.args.attention_schedule.iter())
            .enumerate()
        {
            let cache = cache.as_ref().ok_or_else(|| {
                Exception::custom(format!("Llama cache is missing layer {layer}"))
            })?;
            let expected = policy.window().map(|window| {
                i32::try_from(window.get()).expect("validated Llama attention window fits i32")
            });
            if cache.max_size() != expected {
                return Err(Exception::custom(format!(
                    "Llama cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}",
                    cache.max_size()
                )));
            }
        }
        Ok(())
    }

    /// Forward pass that reports activations to an observer.
    pub fn forward_with_observer<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
        observer: &mut impl ActivationObserver,
    ) -> Result<Array, Exception>
    where
        C: KeyValueCache,
    {
        self.validate_cache(input.cache)?;
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

impl<C> Module<ModelInput<'_, C>> for ResidentModel
where
    C: KeyValueCache,
{
    type Output = Array;

    type Error = Exception;

    fn forward(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Self::Output, Self::Error> {
        self.validate_cache(input.cache)?;
        let out = self.model.forward(input, stream)?;
        project_logits_maybe_quantized(
            &mut self.lm_head,
            &mut self.model.embed_tokens,
            &out,
            stream,
        )
    }

    fn training_mode(&mut self, mode: bool) {
        <ResidentDecoder as Module<ModelInput<'_, C>>>::training_mode(&mut self.model, mode);
        if let Some(lm_head) = &mut self.lm_head {
            lm_head.training_mode(mode);
        }
    }
}

/// Loads `tokenizer.json` from a Llama model directory.
pub fn load_llama_tokenizer(model_dir: impl AsRef<Path>) -> Result<Tokenizer, Error> {
    let file = model_dir.as_ref().join("tokenizer.json");
    Tokenizer::from_file(file).map_err(Into::into)
}

/// Reads and normalizes Llama model arguments from `config.json`.
pub fn get_llama_model_args(model_dir: impl AsRef<Path>) -> Result<ModelArgs, Error> {
    let model_args_filename = model_dir.as_ref().join("config.json");
    let file = std::fs::File::open(model_args_filename)?;
    let source: ModelArgsSource = serde_json::from_reader(file)?;
    normalize_model_args(source)
}

fn normalize_model_args(mut source: ModelArgsSource) -> Result<ModelArgs, Error> {
    if source.num_key_value_heads == 0 {
        source.num_key_value_heads = source.num_attention_heads;
    }
    if source.head_dim == 0 {
        if source.num_attention_heads <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "num_attention_heads must be positive, got {}",
                source.num_attention_heads
            )));
        }
        source.head_dim = source.hidden_size / source.num_attention_heads;
    }
    if source.max_position_embeddings == 0 {
        source.max_position_embeddings = 2048;
    }
    let layer_count = usize::try_from(source.num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "num_hidden_layers must be positive, got {}",
            source.num_hidden_layers
        ))
    })?;
    let attention_schedule = match normalize_hf_sliding_window(source.sliding_window)? {
        None => LayerSchedule::all_full(layer_count),
        Some(window) => LayerSchedule::all_sliding(layer_count, window),
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let model_args = ModelArgs {
        model_type: source.model_type,
        hidden_size: source.hidden_size,
        num_hidden_layers: source.num_hidden_layers,
        intermediate_size: source.intermediate_size,
        num_attention_heads: source.num_attention_heads,
        rms_norm_eps: source.rms_norm_eps,
        vocab_size: source.vocab_size,
        num_key_value_heads: source.num_key_value_heads,
        max_position_embeddings: source.max_position_embeddings,
        rope_theta: source.rope_theta,
        rope_traditional: source.rope_traditional,
        head_dim: source.head_dim,
        tie_word_embeddings: source.tie_word_embeddings,
        attention_bias: source.attention_bias,
        mlp_bias: source.mlp_bias,
        rope_scaling: source.rope_scaling,
        attention_schedule,
        quantization: source.quantization,
        quantization_config: source.quantization_config,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    validate_model_args(&model_args)?;
    Ok(model_args)
}

fn normalize_hf_sliding_window(value: Option<Value>) -> Result<Option<u32>, Error> {
    let Some(value) = value else {
        return Ok(None);
    };
    let Value::Number(number) = value else {
        return Err(Error::UnsupportedArchitecture(format!(
            "sliding_window must be a positive integer or null, got {value}"
        )));
    };
    if let Some(window) = number.as_u64() {
        let window = u32::try_from(window).map_err(|_| {
            Error::UnsupportedArchitecture(format!("sliding_window exceeds u32: {window}"))
        })?;
        if window == 0 {
            return Err(Error::UnsupportedArchitecture(
                "sliding_window must be positive, got 0".into(),
            ));
        }
        return Ok(Some(window));
    }
    if let Some(window) = number.as_i64() {
        return Err(Error::UnsupportedArchitecture(format!(
            "sliding_window must be positive, got {window}"
        )));
    }
    Err(Error::UnsupportedArchitecture(format!(
        "sliding_window must use an integer encoding, got {number}"
    )))
}

fn validate_model_args(model_args: &ModelArgs) -> Result<(), Error> {
    if !matches!(model_args.model_type.as_str(), "llama" | "mistral") {
        return Err(Error::UnsupportedModelType(model_args.model_type.clone()));
    }
    for (name, value) in [
        ("hidden_size", model_args.hidden_size),
        ("num_hidden_layers", model_args.num_hidden_layers),
        ("intermediate_size", model_args.intermediate_size),
        ("num_attention_heads", model_args.num_attention_heads),
        ("num_key_value_heads", model_args.num_key_value_heads),
        ("vocab_size", model_args.vocab_size),
        (
            "max_position_embeddings",
            model_args.max_position_embeddings,
        ),
        ("head_dim", model_args.head_dim),
    ] {
        if value <= 0 {
            return Err(Error::UnsupportedArchitecture(format!(
                "{name} must be positive, got {value}"
            )));
        }
    }
    if model_args.num_attention_heads % model_args.num_key_value_heads != 0 {
        return Err(Error::UnsupportedArchitecture(format!(
            "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
            model_args.num_attention_heads, model_args.num_key_value_heads
        )));
    }
    for (name, heads) in [
        ("query projection", model_args.num_attention_heads),
        ("key/value projection", model_args.num_key_value_heads),
    ] {
        heads.checked_mul(model_args.head_dim).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "{name} width overflows i32: {heads} heads x head_dim {}",
                model_args.head_dim
            ))
        })?;
    }
    if model_args.attention_schedule.len() != model_args.num_hidden_layers as usize {
        return Err(Error::UnsupportedArchitecture(format!(
            "Llama attention schedule has {} layers, expected {}",
            model_args.attention_schedule.len(),
            model_args.num_hidden_layers
        )));
    }
    for window in model_args.attention_schedule.sliding_windows().keys() {
        if i32::try_from(window.get()).is_err() {
            return Err(Error::UnsupportedArchitecture(format!(
                "Llama sliding attention window {} exceeds i32",
                window.get()
            )));
        }
    }
    validate_rope_scaling_config(&model_args.rope_scaling)?;
    Ok(())
}

pub(crate) fn validate_model_config_value(config: &Value) -> Result<(), Error> {
    model_args_from_config_value(config).map(|_| ())
}

/// Parses and normalizes a Hugging Face Llama/Mistral configuration value.
///
/// Raw `sliding_window` metadata is consumed once and replaced by the exact
/// per-layer [`LayerSchedule<AttentionPolicy>`] stored in [`ModelArgs`].
pub fn model_args_from_config_value(config: &Value) -> Result<ModelArgs, Error> {
    let args = serde_json::from_value::<ModelArgsSource>(config.clone()).map_err(|error| {
        Error::UnsupportedArchitecture(format!("invalid Llama-compatible config: {error}"))
    })?;
    normalize_model_args(args)
}

pub(crate) struct PreparedLlamaGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) fn prepare_llama_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    _weights_stream: &Stream,
) -> Result<PreparedLlamaGguf, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    if !matches!(architecture.as_str(), "llama" | "mistral") {
        return Err(Error::UnsupportedArchitecture(format!(
            "GGUF architecture {architecture:?}; this loader supports llama and mistral"
        )));
    }
    let gguf_architecture = crate::core::GgufArchitecture::resolve(&architecture)?;
    crate::backend::mlx::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::backend::mlx::ModelLoadOptions::default(),
    )
    .into_loader_result()?;

    checkpoint
        .catalog()
        .translated_outputs(translate_gguf_weight_name)
        .map_err(safemlx::error::IoError::from)?;
    let mut args = model_args_from_gguf_catalog(checkpoint, metadata)?;
    let quantized_weight_configs =
        gguf_quantization_configs(checkpoint, translate_gguf_weight_name)?;
    if let Some(quantization) = quantization {
        args.quantized_weights = None;
        args.quantization = Some(quantization);
        args.quantized_weight_configs = None;
    } else {
        args.quantized_weights = Some(quantized_weight_configs.keys().cloned().collect());
        args.quantization = None;
        args.quantized_weight_configs = Some(quantized_weight_configs);
    }

    let eos_token_ids = crate::backend::mlx::gguf_eos_token_ids(metadata)?;
    Ok(PreparedLlamaGguf {
        args,
        eos_token_ids,
    })
}

/// Parses the GGUF arguments shared by structural preflight and loading.
pub(crate) fn model_args_from_gguf_catalog(
    arrays: &impl GgufTensorNames,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> Result<ModelArgs, Error> {
    let architecture = gguf_string(metadata, "general.architecture")?;
    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let hidden_size = gguf_i32(metadata, &key("embedding_length"))?;
    let num_attention_heads = gguf_i32(metadata, &key("attention.head_count"))?;
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
    let rope_theta =
        gguf_optional_f32(metadata, &key("rope.freq_base"))?.unwrap_or_else(default_rope_theta);
    let rope_scaling = gguf_rope_scaling(metadata, &architecture)?;
    let sliding_window = gguf_optional_i64(metadata, &key("attention.sliding_window"))?;
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
        None => gguf_i32(metadata, &key("vocab_size"))?,
    };

    let num_hidden_layers = gguf_i32(metadata, &key("block_count"))?;
    let layer_count = usize::try_from(num_hidden_layers).map_err(|_| {
        Error::UnsupportedArchitecture(format!(
            "GGUF block count must be positive, got {num_hidden_layers}"
        ))
    })?;
    // GGUF defines zero as disabled for this scalar; positive values enable the
    // same exact window on every decoder layer.
    let attention_schedule = match sliding_window {
        None | Some(0) => LayerSchedule::all_full(layer_count),
        Some(window) => {
            let window = u32::try_from(window).map_err(|_| {
                Error::UnsupportedArchitecture(format!(
                    "GGUF sliding-window size must be positive and fit u32, got {window}"
                ))
            })?;
            LayerSchedule::all_sliding(layer_count, window)
        }
    }
    .map_err(|error| Error::UnsupportedArchitecture(error.to_string()))?;
    let args = ModelArgs {
        model_type: architecture.clone(),
        hidden_size,
        num_hidden_layers,
        intermediate_size: gguf_i32(metadata, &key("feed_forward_length"))?,
        num_attention_heads,
        rms_norm_eps: gguf_f32(metadata, &key("attention.layer_norm_rms_epsilon"))?,
        vocab_size,
        num_key_value_heads,
        max_position_embeddings: gguf_i32(metadata, &key("context_length"))?,
        rope_theta,
        rope_traditional: true,
        head_dim,
        tie_word_embeddings: !arrays.contains_gguf_tensor("output.weight"),
        attention_bias: arrays.any_gguf_tensor(|name| {
            name.starts_with("blk.")
                && matches!(
                    name.rsplit_once('.'),
                    Some((prefix, "bias")) if prefix.ends_with("attn_q")
                        || prefix.ends_with("attn_k")
                        || prefix.ends_with("attn_v")
                        || prefix.ends_with("attn_output")
                )
        }),
        mlp_bias: arrays.any_gguf_tensor(|name| {
            name.starts_with("blk.")
                && matches!(
                    name.rsplit_once('.'),
                    Some((prefix, "bias")) if prefix.ends_with("ffn_gate")
                        || prefix.ends_with("ffn_down")
                        || prefix.ends_with("ffn_up")
                )
        }),
        rope_scaling,
        attention_schedule,
        quantization: None,
        quantization_config: None,
        quantized_weights: None,
        quantized_weight_configs: None,
    };
    validate_model_args(&args)?;
    Ok(args)
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
            let scaling_factor_key = format!("{architecture}.rope.scaling.factor");
            let factor = gguf_optional_f32(metadata, &scaling_factor_key)?.ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "linear GGUF RoPE scaling is missing {scaling_factor_key}"
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
        other => Err(Error::UnsupportedArchitecture(format!(
            "GGUF RoPE scaling type {other:?} is not supported by the initial GGUF loader"
        ))),
    }
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    name.replace("blk.", "model.layers.")
        .replace("ffn_gate", "mlp.gate_proj")
        .replace("ffn_down", "mlp.down_proj")
        .replace("ffn_up", "mlp.up_proj")
        .replace("attn_q", "self_attn.q_proj")
        .replace("attn_k", "self_attn.k_proj")
        .replace("attn_v", "self_attn.v_proj")
        .replace("attn_output", "self_attn.o_proj")
        .replace("attn_norm", "input_layernorm")
        .replace("ffn_norm", "post_attention_layernorm")
        .replace("token_embd", "model.embed_tokens")
        .replace("output_norm", "model.norm")
        .replace("output", "lm_head")
}

fn gguf_string(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<String, Error> {
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

fn gguf_i32(metadata: &HashMap<String, GgufMetadataValue>, key: &str) -> Result<i32, Error> {
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

#[derive(Debug, Clone, Deserialize)]
/// Hugging Face safetensors index file.
pub struct WeightMap {
    /// Index metadata.
    pub metadata: HashMap<String, Value>,
    /// Mapping from tensor name to shard file name.
    pub weight_map: HashMap<String, String>,
}

impl<C> CausalLm<Vec<Option<C>>> for ResidentModel
where
    C: KeyValueCache,
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

/// Llama token generation iterator.
pub type Generate<'a, C, S = crate::backend::mlx::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, ResidentModel, Vec<Option<C>>, S>;
