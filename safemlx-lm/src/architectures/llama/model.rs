//! Llama decoder-only model implementation.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    macros::{ModuleParameters, Quantizable},
    module::Module,
    nn,
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    quantization::MaybeQuantized,
    Array, Dtype, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use crate::nn::generation::sample;

#[cfg(test)]
use crate::runtime::checkpoint::load::{
    gguf_metadata, load_gguf_strict, load_safetensors_dir_lenient, StrictLoadConfig,
    StrictLoadReport,
};
use crate::{
    api::{
        common::{
            self,
            attention::{
                apply_rope_and_update_cache, attention_probabilities, batch_seq, finish_attention,
                reshape_attention_projection,
            },
            generation::CausalLm,
            layers::SwiGluMlp,
            linear::project_logits_maybe_quantized,
        },
        input,
    },
    error::Error,
    nn::tensor::{
        create_attention_mask,
        rope::{initialize_rope, FloatOrString, RopeVariant},
        AttentionMask,
    },
    runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    runtime::checkpoint::quantization::WeightQuantization,
    runtime::execution::inspection::ActivationObserver,
    runtime::{
        attention::{AttentionPolicy, LayerSchedule},
        cache::residency::derive_prompt_cache_architecture_fingerprint,
        cache::{ConcatKeyValueCache, KeyValueCache},
    },
};
#[cfg(test)]
use safemlx::module::ModuleParametersExt;

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
        let attention = crate::nn::parallel::forward_row_parallel(
            &mut attention.o_proj,
            &attended,
            group,
            stream,
        )?;
        let hidden = hidden.add(attention, stream)?;
        let normalized = self.post_attention_layernorm.forward(&hidden, stream)?;
        let gate =
            crate::nn::layers::silu(self.mlp.gate_proj.forward(&normalized, stream)?, stream)?;
        let up = self.mlp.up_proj.forward(&normalized, stream)?;
        let mlp = crate::nn::parallel::forward_row_parallel(
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

#[cfg(test)]
pub(crate) struct LoadedLlamaGguf {
    pub(crate) model: ResidentModel,
    pub(crate) eos_token_ids: Vec<u32>,
}

pub(crate) struct PreparedLlamaGguf {
    pub(crate) args: ModelArgs,
    pub(crate) eos_token_ids: Vec<u32>,
}

/// Loads a Llama-compatible GGUF checkpoint, including Mistral.
///
/// Dense tensors and GGUF Q2_K, Q3_K, Q4_0, Q4_1, Q4_K, Q5_K, Q6_K, and Q8_0 tensors are
/// supported. Quantized formats are consumed in the packed affine
/// representation emitted by MLX's GGUF loader.
#[cfg(test)]
pub fn load_llama_gguf(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ResidentModel, Error> {
    Ok(load_llama_gguf_with_metadata(gguf_file, stream, weights_stream)?.model)
}

#[cfg(test)]
pub(crate) fn load_llama_gguf_with_metadata(
    gguf_file: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedLlamaGguf, Error> {
    let gguf_file = gguf_file.as_ref();
    let checkpoint = GgufCheckpoint::open(gguf_file)?;
    let metadata = gguf_metadata(&checkpoint);
    load_llama_gguf_checkpoint(&checkpoint, metadata, None, stream, weights_stream)
}

#[cfg(test)]
pub(crate) fn load_llama_gguf_checkpoint(
    checkpoint: &GgufCheckpoint,
    metadata: HashMap<String, GgufMetadataValue>,
    quantization: Option<WeightQuantization>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<LoadedLlamaGguf, Error> {
    let prepared =
        prepare_llama_gguf_checkpoint(checkpoint, &metadata, quantization, weights_stream)?;
    let mut model = ResidentModel::new(prepared.args, stream)?;
    let config = StrictLoadConfig::default().allow_unused_prefix("rope_freqs.");
    let mut report = StrictLoadReport::default();
    load_gguf_strict(
        &mut model,
        checkpoint,
        quantization.map(|value| (value, stream)),
        &config,
        &mut report,
        |name, value| Ok((translate_gguf_weight_name(&name), value)),
    )?;
    report.finish(&model, &config)?;
    model.copy_to_stream(stream)?;

    Ok(LoadedLlamaGguf {
        model,
        eos_token_ids: prepared.eos_token_ids,
    })
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
    let gguf_architecture = crate::api::GgufArchitecture::resolve(&architecture)?;
    crate::api::structural::validate_gguf(
        gguf_architecture,
        checkpoint,
        metadata,
        crate::api::ModelLoadOptions::default(),
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

    let eos_token_ids = crate::api::gguf_eos_token_ids(metadata)?;
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

/// Test-only eager reference loader used to compare the canonical engine.
#[cfg(test)]
pub(crate) fn load_test_resident_llama_model(
    model_dir: impl AsRef<Path>,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<ResidentModel, Error> {
    let model_dir = model_dir.as_ref();
    crate::api::structural::validate_safetensors_load_path(
        crate::api::ModelKind::Llama,
        model_dir,
        crate::api::ModelLoadOptions::default(),
    )?;
    let model_args = get_llama_model_args(model_dir)?;
    let mut model = ResidentModel::new(model_args, stream)?;

    load_safetensors_dir_lenient(&mut model, model_dir, weights_stream)?;
    model.copy_to_stream(stream)?;

    Ok(model)
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
pub type Generate<'a, C, S = crate::runtime::generation::sampler::DefaultSampler> =
    common::generation::Generate<'a, ResidentModel, Vec<Option<C>>, S>;

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, HashSet},
        env::home_dir,
        fs,
    };

    use lazy_static::lazy_static;
    use safemlx::{
        module::{Module, ModuleParameters},
        ops::indexing::{NewAxis, TryIndexOp},
        ops::{GgufMetadataArray, GgufMetadataValue},
        transforms::eval,
        Array,
    };

    use crate::{
        architectures::llama::model::{load_llama_tokenizer, load_test_resident_llama_model},
        runtime::cache::{ConcatKeyValueCache, KeyValueCache},
        runtime::checkpoint::quantization::AffineQuantization,
    };

    #[test]
    fn normalizes_hermes_mistral_config() {
        let args = super::model_args_from_config_value(&serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "intermediate_size": 14336,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 0.00001,
            "vocab_size": 32032,
            "max_position_embeddings": 32768,
            "rope_theta": 10000.0,
            "sliding_window": 4096,
            "tie_word_embeddings": false
        }))
        .unwrap();

        assert_eq!(args.model_type, "mistral");
        assert_eq!(args.head_dim, 128);
        assert_eq!(args.num_key_value_heads, 8);
        assert_eq!(args.attention_schedule.sliding_layer_count(), 32);
        assert_eq!(
            args.attention_schedule
                .get(0)
                .unwrap()
                .window()
                .unwrap()
                .get(),
            4096
        );
    }

    fn tiny_config(sliding_window: Option<serde_json::Value>) -> serde_json::Value {
        let mut config = serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 32,
            "num_hidden_layers": 4,
            "intermediate_size": 64,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "rms_norm_eps": 0.00001,
            "vocab_size": 64,
            "max_position_embeddings": 128
        });
        if let Some(window) = sliding_window {
            config["sliding_window"] = window;
        }
        config
    }

    #[test]
    fn hf_llama_and_mistral_normalize_to_exact_attention_schedules() {
        let full = super::model_args_from_config_value(&tiny_config(None)).unwrap();
        assert_eq!(full.attention_schedule.full_layer_count(), 4);
        assert_eq!(full.attention_schedule.sliding_layer_count(), 0);

        let sliding = super::model_args_from_config_value(&tiny_config(Some(7.into()))).unwrap();
        assert_eq!(sliding.attention_schedule.full_layer_count(), 0);
        assert_eq!(sliding.attention_schedule.sliding_layer_count(), 4);
        assert!(sliding
            .attention_schedule
            .iter()
            .all(|policy| policy.window().unwrap().get() == 7));
    }

    #[test]
    fn hf_mistral_rejects_invalid_and_overflowing_windows() {
        for invalid in [
            serde_json::json!(0),
            serde_json::json!(-1),
            serde_json::json!(4_294_967_296u64),
        ] {
            let error = super::model_args_from_config_value(&tiny_config(Some(invalid)))
                .unwrap_err()
                .to_string();
            assert!(error.contains("sliding_window") || error.contains("window"));
        }
    }

    #[test]
    fn fingerprint_and_cache_geometry_include_ordered_attention_schedule() {
        use crate::runtime::attention::{AttentionPolicy, LayerSchedule};
        use crate::runtime::cache::KeyValueCache;

        let mut first = super::model_args_from_config_value(&tiny_config(None)).unwrap();
        first.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        let mut second = first.clone();
        second.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::Full,
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        assert_ne!(
            super::prompt_cache_architecture_fingerprint(&first),
            super::prompt_cache_architecture_fingerprint(&second)
        );

        let context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let model = super::ResidentModel::new(first, context.stream()).unwrap();
        let cache = model.new_cache();
        assert_eq!(
            cache
                .iter()
                .map(|cache| cache.as_ref().unwrap().max_size())
                .collect::<Vec<_>>(),
            vec![Some(3), None, Some(5), None]
        );
    }

    fn initialize_model(module: &mut impl ModuleParameters, stream: &safemlx::Stream) {
        let mut names = module
            .parameters()
            .flatten()
            .keys()
            .map(ToString::to_string)
            .collect::<Vec<_>>();
        names.sort();
        let mut parameters = module.parameters_mut().flatten();
        for (index, name) in names.iter().enumerate() {
            let parameter = parameters.get_mut(name.as_str()).unwrap();
            let shape = parameter.shape().to_vec();
            let dtype = parameter.dtype();
            **parameter =
                Array::full::<f32>(&shape, Array::from_f32(0.001 * (index + 1) as f32), stream)
                    .unwrap()
                    .as_dtype(dtype, stream)
                    .unwrap();
        }
    }

    fn assert_close(left: &Array, right: &Array, stream: &safemlx::Stream) {
        assert!(left
            .all_close(right, Some(3e-5), Some(3e-5), None, stream)
            .unwrap()
            .item::<bool>(stream));
    }

    #[test]
    fn arbitrary_schedule_prefill_decode_and_paged_cache_parity() {
        use crate::runtime::{
            attention::{AttentionPolicy, LayerSchedule},
            cache::{
                residency::{CacheResidencyManager, PagedCacheOptions},
                PagedKeyValueCache,
            },
        };

        let context =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = context.stream();
        let mut args = super::model_args_from_config_value(&tiny_config(None)).unwrap();
        args.attention_schedule = LayerSchedule::new(
            4,
            vec![
                AttentionPolicy::sliding(3).unwrap(),
                AttentionPolicy::Full,
                AttentionPolicy::sliding(5).unwrap(),
                AttentionPolicy::Full,
            ],
        )
        .unwrap();
        let mut ordinary = super::ResidentModel::new(args, stream).unwrap();
        initialize_model(&mut ordinary, stream);
        let mut paged = ordinary.clone();
        let mut tokenwise = ordinary.clone();
        let mut ordinary_cache = ordinary.new_cache();
        let mut tokenwise_cache = tokenwise.new_cache();
        let manager = CacheResidencyManager::new(
            PagedCacheOptions::new(2, 1 << 20, 1 << 20, 1)
                .unwrap()
                .with_full_attention(true),
        )
        .unwrap();
        let mut paged_cache = paged
            .args
            .attention_schedule
            .iter()
            .enumerate()
            .map(|(layer, policy)| {
                PagedKeyValueCache::new(
                    manager.clone(),
                    layer,
                    policy.window().map(|window| window.get() as i32),
                )
                .map(Some)
                .unwrap()
            })
            .collect::<Vec<_>>();
        let prompt = Array::from_slice(&[1u32, 2, 3, 4], &[1, 4]);
        let expected = ordinary
            .forward(
                super::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: &mut ordinary_cache,
                },
                stream,
            )
            .unwrap();
        let actual = paged
            .forward(
                super::ModelInput {
                    inputs: &prompt,
                    mask: None,
                    cache: &mut paged_cache,
                },
                stream,
            )
            .unwrap();
        assert_close(&expected, &actual, stream);

        let mut decoded = None;
        for token in [1u32, 2, 3, 4] {
            decoded = Some(
                tokenwise
                    .forward(
                        super::ModelInput {
                            inputs: &Array::from_slice(&[token], &[1, 1]),
                            mask: None,
                            cache: &mut tokenwise_cache,
                        },
                        stream,
                    )
                    .unwrap(),
            );
        }
        let expected_last = expected.try_index_device((.., -1, ..), stream).unwrap();
        let decoded_last = decoded
            .unwrap()
            .try_index_device((.., -1, ..), stream)
            .unwrap();
        assert_close(&expected_last, &decoded_last, stream);
        assert_eq!(
            ordinary_cache
                .iter()
                .map(|cache| cache.as_ref().unwrap().retained_arrays()[0].dim(-2))
                .collect::<Vec<_>>(),
            vec![2, 4, 4, 4]
        );
    }

    #[test]
    fn prompt_cache_architecture_fingerprint_is_derived_from_rope_configuration() {
        let mut args = super::model_args_from_config_value(&serde_json::json!({
            "model_type": "llama",
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "intermediate_size": 128,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
            "rms_norm_eps": 0.00001,
            "vocab_size": 128,
            "max_position_embeddings": 4096,
            "rope_theta": 10000.0,
            "rope_scaling": {"factor": 2.0, "rope_type": "linear"}
        }))
        .unwrap();
        let first = super::prompt_cache_architecture_fingerprint(&args);
        assert_eq!(first, super::prompt_cache_architecture_fingerprint(&args));
        args.rope_theta = 500_000.0;
        let changed = super::prompt_cache_architecture_fingerprint(&args);
        assert_ne!(first, changed);
    }

    #[test]
    fn llama_fingerprint_and_runtime_preserve_distinct_ordered_windows() {
        let mut args = super::model_args_from_config_value(&serde_json::json!({
            "model_type": "llama", "hidden_size": 64, "num_hidden_layers": 4,
            "intermediate_size": 128, "num_attention_heads": 4,
            "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 0.00001,
            "vocab_size": 128, "max_position_embeddings": 4096,
            "rope_theta": 10000.0
        }))
        .unwrap();
        args.attention_schedule = crate::runtime::attention::LayerSchedule::new(
            4,
            vec![
                crate::runtime::attention::AttentionPolicy::Full,
                crate::runtime::attention::AttentionPolicy::sliding(3).unwrap(),
                crate::runtime::attention::AttentionPolicy::Full,
                crate::runtime::attention::AttentionPolicy::sliding(9).unwrap(),
            ],
        )
        .unwrap();
        super::validate_model_args(&args).unwrap();
        assert_eq!(
            args.attention_schedule
                .iter()
                .map(|policy| policy.window().map(|window| window.get() as i32))
                .collect::<Vec<_>>(),
            vec![None, Some(3), None, Some(9)]
        );
        let first = super::prompt_cache_architecture_fingerprint(&args);
        args.attention_schedule = crate::runtime::attention::LayerSchedule::new(
            4,
            vec![
                crate::runtime::attention::AttentionPolicy::sliding(3).unwrap(),
                crate::runtime::attention::AttentionPolicy::Full,
                crate::runtime::attention::AttentionPolicy::Full,
                crate::runtime::attention::AttentionPolicy::sliding(9).unwrap(),
            ],
        )
        .unwrap();
        let reordered = super::prompt_cache_architecture_fingerprint(&args);
        assert_ne!(first, reordered);
    }

    #[test]
    fn preserves_mistral_small_explicit_head_dimension() {
        let args = super::model_args_from_config_value(&serde_json::json!({
            "model_type": "mistral",
            "hidden_size": 5120,
            "num_hidden_layers": 40,
            "intermediate_size": 32768,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 0.00001,
            "vocab_size": 131072,
            "max_position_embeddings": 32768,
            "rope_theta": 100000000.0,
            "sliding_window": null,
            "tie_word_embeddings": false
        }))
        .unwrap();

        assert_eq!(args.head_dim, 128);
        assert_eq!(args.hidden_size, 5120);
        assert_eq!(args.attention_schedule.full_layer_count(), 40);
    }

    #[test]
    fn translates_gguf_llama_weight_names() {
        assert_eq!(
            super::translate_gguf_weight_name("blk.3.attn_q.weight"),
            "model.layers.3.self_attn.q_proj.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("blk.1.ffn_down.scales"),
            "model.layers.1.mlp.down_proj.scales"
        );
        assert_eq!(
            super::translate_gguf_weight_name("token_embd.weight"),
            "model.embed_tokens.weight"
        );
        assert_eq!(
            super::translate_gguf_weight_name("output_norm.weight"),
            "model.norm.weight"
        );
    }

    #[test]
    fn loads_dense_mistral_from_synthetic_gguf_checkpoint() {
        use safemlx::module::ModuleParameters;

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let stream = ctx.stream();
        let source = super::ResidentModel::new(
            super::ModelArgs {
                model_type: "mistral".into(),
                hidden_size: 32,
                num_hidden_layers: 1,
                intermediate_size: 256,
                num_attention_heads: 1,
                rms_norm_eps: 1e-5,
                vocab_size: 32,
                num_key_value_heads: 1,
                max_position_embeddings: 128,
                rope_theta: 10_000.0,
                rope_traditional: true,
                head_dim: 32,
                tie_word_embeddings: true,
                attention_bias: false,
                mlp_bias: false,
                rope_scaling: None,
                attention_schedule: crate::runtime::attention::LayerSchedule::all_sliding(1, 16)
                    .unwrap(),
                quantization: None,
                quantization_config: None,
                quantized_weights: None,
                quantized_weight_configs: None,
            },
            stream,
        )
        .unwrap();
        let arrays: HashMap<String, Array> = source
            .parameters()
            .flatten()
            .into_iter()
            .map(|(name, value)| {
                let name = name
                    .replace("model.layers.", "blk.")
                    .replace("self_attn.q_proj", "attn_q")
                    .replace("self_attn.k_proj", "attn_k")
                    .replace("self_attn.v_proj", "attn_v")
                    .replace("self_attn.o_proj", "attn_output")
                    .replace("input_layernorm", "attn_norm")
                    .replace("post_attention_layernorm", "ffn_norm")
                    .replace("mlp.gate_proj", "ffn_gate")
                    .replace("mlp.down_proj", "ffn_down")
                    .replace("mlp.up_proj", "ffn_up")
                    .replace("model.embed_tokens", "token_embd")
                    .replace("model.norm", "output_norm");
                (name, value.clone())
            })
            .collect();
        let metadata = HashMap::from([
            (
                "general.architecture".into(),
                GgufMetadataValue::String("mistral".into()),
            ),
            (
                "mistral.embedding_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            ("mistral.block_count".into(), GgufMetadataValue::Uint32(1)),
            (
                "mistral.feed_forward_length".into(),
                GgufMetadataValue::Uint32(256),
            ),
            (
                "mistral.attention.head_count".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "mistral.attention.head_count_kv".into(),
                GgufMetadataValue::Uint32(1),
            ),
            (
                "mistral.attention.key_length".into(),
                GgufMetadataValue::Uint32(32),
            ),
            (
                "mistral.attention.layer_norm_rms_epsilon".into(),
                GgufMetadataValue::Float32(1e-5),
            ),
            (
                "mistral.attention.sliding_window".into(),
                GgufMetadataValue::Uint32(16),
            ),
            (
                "mistral.context_length".into(),
                GgufMetadataValue::Uint32(128),
            ),
            (
                "mistral.rope.freq_base".into(),
                GgufMetadataValue::Float32(10_000.0),
            ),
            (
                "tokenizer.ggml.tokens".into(),
                GgufMetadataValue::Array(GgufMetadataArray::String(vec!["token".into(); 32])),
            ),
            (
                "tokenizer.ggml.eos_token_id".into(),
                GgufMetadataValue::Uint32(2),
            ),
        ]);

        let fixture = crate::test_utils::SyntheticGguf::dense(&arrays, &metadata);
        let loaded = super::load_llama_gguf_with_metadata(fixture.path(), stream, stream).unwrap();

        assert_eq!(loaded.model.model_type(), "mistral");
        assert_eq!(
            loaded
                .model
                .args
                .attention_schedule
                .get(0)
                .unwrap()
                .window()
                .unwrap()
                .get(),
            16
        );
        assert_eq!(loaded.eos_token_ids, vec![2]);

        let mut zero_window = metadata.clone();
        zero_window.insert(
            "mistral.attention.sliding_window".into(),
            GgufMetadataValue::Uint32(0),
        );
        let zero_args = super::model_args_from_gguf_catalog(&arrays, &zero_window).unwrap();
        assert_eq!(zero_args.attention_schedule.full_layer_count(), 1);

        let mut negative_window = metadata.clone();
        negative_window.insert(
            "mistral.attention.sliding_window".into(),
            GgufMetadataValue::Int32(-1),
        );
        assert!(
            super::model_args_from_gguf_catalog(&arrays, &negative_window)
                .unwrap_err()
                .to_string()
                .contains("sliding-window")
        );

        let mut wrong_window = metadata.clone();
        wrong_window.insert(
            "mistral.attention.sliding_window".into(),
            GgufMetadataValue::String("16".into()),
        );
        assert!(super::model_args_from_gguf_catalog(&arrays, &wrong_window)
            .unwrap_err()
            .to_string()
            .contains("wrong type"));

        // A model-loader smoke fixture mixing ordinary dense tensors, one
        // affine K-quant, and one nonlinear IQ tensor. Q2_K uses MLX's packed
        // affine layout while IQ4_NL retains its original GGML blocks.
        let mixed_fixture =
            crate::test_utils::SyntheticGguf::with_packed_tensors(&arrays, &metadata, |name, _| {
                match name {
                    "blk.0.attn_q.weight" => Some(safemlx_gguf::GgmlType::IQ4NL),
                    "blk.0.ffn_down.weight" => Some(safemlx_gguf::GgmlType::Q2K),
                    _ => None,
                }
            });
        let checkpoint = safemlx::ops::GgufCheckpoint::open(mixed_fixture.path()).unwrap();
        let mut mixed = super::load_llama_gguf_checkpoint(
            &checkpoint,
            crate::runtime::checkpoint::load::gguf_metadata(&checkpoint),
            None,
            stream,
            stream,
        )
        .unwrap();
        let mixed_params = mixed.model.parameters().flatten();
        assert!(mixed_params.contains_key("model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(mixed_params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(mixed_params.contains_key("model.layers.0.mlp.down_proj.inner.weight"));
        assert!(mixed_params.contains_key("model.layers.0.mlp.down_proj.scales"));
        drop(mixed_params);

        let safemlx::quantization::MaybeQuantized::Quantized(q_proj) =
            &mut mixed.model.model.layers[0].self_attn.q_proj
        else {
            panic!("IQ4_NL projection must use a quantized module");
        };
        assert_eq!(
            q_proj.native_format,
            Some(safemlx::native_quantization::NativeQuantizationFormat::GgufIQ4NL)
        );
        assert_eq!(q_proj.inner.weight.value.dtype(), safemlx::Dtype::Uint8);
        assert_eq!(q_proj.inner.weight.value.shape(), &[32, 18]);
        let projected = q_proj
            .forward(&Array::from_slice(&[1.0f32; 32], &[1, 32]), stream)
            .unwrap();
        eval([&projected]).unwrap();
        assert_eq!(projected.shape(), &[1, 32]);

        let gpu = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let checkpoint = safemlx::ops::GgufCheckpoint::open(fixture.path()).unwrap();
        let metadata = crate::runtime::checkpoint::load::gguf_metadata(&checkpoint);
        let quantized = super::load_llama_gguf_checkpoint(
            &checkpoint,
            metadata,
            Some(crate::runtime::checkpoint::quantization::WeightQuantization::MxFp4),
            gpu.stream(),
            stream,
        )
        .unwrap();
        let params = quantized.model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(!params.contains_key("model.layers.0.self_attn.q_proj.biases"));
        assert!(params.contains_key("model.embed_tokens.scales"));
        assert!(!params.contains_key("model.embed_tokens.biases"));
    }

    #[test]
    fn mixed_quantization_builds_only_selected_llama_parameters() {
        use safemlx::module::ModuleParameters;

        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let selected = HashSet::from(["model.layers.0.self_attn.q_proj.weight".to_string()]);
        let args = super::ModelArgs {
            model_type: "llama".into(),
            hidden_size: 32,
            num_hidden_layers: 1,
            intermediate_size: 32,
            num_attention_heads: 1,
            rms_norm_eps: 1e-5,
            vocab_size: 32,
            num_key_value_heads: 1,
            max_position_embeddings: 128,
            rope_theta: 10_000.0,
            rope_traditional: true,
            head_dim: 32,
            tie_word_embeddings: true,
            attention_bias: false,
            mlp_bias: false,
            rope_scaling: None,
            attention_schedule: crate::runtime::attention::LayerSchedule::all_full(1).unwrap(),
            quantization: Some(AffineQuantization::new(32, 4).unwrap().into()),
            quantization_config: None,
            quantized_weights: Some(selected),
            quantized_weight_configs: None,
        };
        let model = super::ResidentModel::new(args, ctx.stream()).unwrap();
        let params = model.parameters().flatten();
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.inner.weight"));
        assert!(params.contains_key("model.layers.0.self_attn.q_proj.scales"));
        assert!(params.contains_key("model.layers.0.self_attn.k_proj.weight"));
        assert!(!params.contains_key("model.layers.0.self_attn.k_proj.scales"));
    }

    /// Resolve the HuggingFace cache directory to the actual snapshot path.
    /// The structure is:
    ///   models--<org>--<name>/
    ///     refs/
    ///       main  (contains the commit hash)
    ///     snapshots/
    ///       <commit_hash>/  (actual model files)
    fn resolve_hf_cache_dir(model_cache_dir: &str) -> String {
        let refs_main = std::path::Path::new(model_cache_dir)
            .join("refs")
            .join("main");
        let commit_hash = fs::read_to_string(&refs_main)
            .unwrap_or_default()
            .trim()
            .to_string();
        std::path::Path::new(model_cache_dir)
            .join("snapshots")
            .join(commit_hash)
            .to_string_lossy()
            .into_owned()
    }

    lazy_static! {
        static ref CACHED_TEST_MODEL_DIR: String = {
            let cache_dir = home_dir()
                .map(|p| {
                    p.join(".cache")
                        .join("huggingface")
                        .join("hub")
                        .join("models--meta-llama--Llama-3.2-1B-Instruct")
                        .to_string_lossy()
                        .into_owned()
                })
                .unwrap_or_default();

            resolve_hf_cache_dir(&cache_dir)
        };
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_llama_model() {
        use safemlx::module::ModuleParameters;

        let model_dir = CACHED_TEST_MODEL_DIR.as_str();
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let model_args = super::get_llama_model_args(model_dir).unwrap();
        let model = super::ResidentModel::new(model_args, stream).unwrap();

        // Print some model parameter keys
        let params = model.parameters().flatten();
        let mut param_keys: Vec<_> = params.keys().map(|k| k.to_string()).collect();
        param_keys.sort();
        println!("=== Model parameter keys (first 20) ===");
        for key in param_keys.iter().take(20) {
            println!("  {key}");
        }

        // Print some safetensor keys
        let weights_path = std::path::Path::new(model_dir).join("model.safetensors");
        let loaded = safemlx::Array::load_safetensors(&weights_path, stream).unwrap();
        let mut weight_keys: Vec<_> = loaded.keys().map(|k| k.to_string()).collect();
        weight_keys.sort();
        println!("=== Safetensor weight keys (first 20) ===");
        for key in weight_keys.iter().take(20) {
            println!("  {key}");
        }

        // Find unmatched keys
        let param_set: std::collections::HashSet<_> = param_keys.iter().collect();
        let weight_set: std::collections::HashSet<_> = weight_keys.iter().collect();
        let unloaded: Vec<_> = weight_set.difference(&param_set).collect();
        let missing: Vec<_> = param_set.difference(&weight_set).collect();
        println!(
            "=== Weight keys NOT in model params ({}) ===",
            unloaded.len()
        );
        for key in unloaded.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "=== Model param keys NOT in weights ({}) ===",
            missing.len()
        );
        for key in missing.iter().take(10) {
            println!("  {key}");
        }
        println!(
            "Total model params: {}, Total weight keys: {}",
            param_keys.len(),
            weight_keys.len()
        );
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_tokenizer() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();

        let _encoding = tokenizer.encode("Hello, world!", true).unwrap();
    }

    #[test]
    #[ignore = "requires local model files"]
    fn test_load_and_run_llama_with_concat_cache() {
        let tokenizer = load_llama_tokenizer(CACHED_TEST_MODEL_DIR.as_str()).unwrap();
        let ctx = safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Gpu, 0));
        let stream = ctx.stream();
        let weights_ctx =
            safemlx::ExecutionContext::new(safemlx::Device::new(safemlx::DeviceType::Cpu, 0));
        let weights_stream = weights_ctx.stream();
        let mut model =
            load_test_resident_llama_model(CACHED_TEST_MODEL_DIR.as_str(), stream, weights_stream)
                .unwrap();

        let prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\nWhat is the capital of France?<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n";
        let encoding = tokenizer.encode(prompt, false).unwrap();
        let prompt_tokens = Array::from(encoding.get_ids())
            .try_index_device(NewAxis, stream)
            .unwrap();
        let mut cache = model.new_cache();

        let eos_token_id = 128001u32;
        let eot_token_id = 128009u32;

        let mut token_ids = Vec::new();
        let input_parts = [crate::runtime::media::input::InputPart::text_token_ids(
            &prompt_tokens,
        )];
        let input = crate::runtime::media::input::ModelInput::new(&input_parts);
        let generate = super::Generate::<ConcatKeyValueCache>::new(
            &mut model, &mut cache, 0.0, input, None, stream,
        );
        for (token, _ntoks) in generate.zip(0..50) {
            let token = token.unwrap();
            eval([&token]).unwrap();
            let token_id = token.item::<u32>(&stream);
            print!("[{}]", token_id);
            if token_id == eos_token_id || token_id == eot_token_id {
                break;
            }
            token_ids.push(token_id);
        }
        println!();

        let output = tokenizer.decode(&token_ids, true).unwrap();
        println!("Response: {output}");
        println!("------");
    }
}
