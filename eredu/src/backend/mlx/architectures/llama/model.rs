//! MLX configuration, checkpoint integration, and resident binding for shared Llama.

use std::{
    collections::{HashMap, HashSet},
    path::Path,
};

use safemlx::{
    error::Exception,
    module::{Module, ModuleParameters},
    ops::indexing::TryIndexOp,
    ops::{GgufCheckpoint, GgufMetadataValue},
    Array, Stream,
};
use serde::Deserialize;
use serde_json::Value;
use tokenizers::Tokenizer;

pub use crate::backend::mlx::nn::generation::sample;

use crate::{
    backend::mlx::error::Error,
    backend::mlx::nn::tensor::{
        create_attention_mask,
        rope::{validate_rope_scaling_config, FloatOrString},
        AttentionMask,
    },
    backend::mlx::nn::{self as common, generation::CausalLm},
    backend::mlx::runtime::cache::{ConcatKeyValueCache, KeyValueCache},
    backend::mlx::runtime::checkpoint::load::{gguf_quantization_configs, GgufTensorNames},
    backend::mlx::runtime::checkpoint::quantization::WeightQuantization,
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

impl eredu_architectures::llama::Config for ModelArgs {
    fn hidden_size(&self) -> i32 {
        self.hidden_size
    }
    fn num_hidden_layers(&self) -> i32 {
        self.num_hidden_layers
    }
    fn intermediate_size(&self) -> i32 {
        self.intermediate_size
    }
    fn num_attention_heads(&self) -> i32 {
        self.num_attention_heads
    }
    fn num_key_value_heads(&self) -> i32 {
        self.num_key_value_heads
    }
    fn head_dim(&self) -> i32 {
        self.head_dim
    }
    fn rms_norm_epsilon(&self) -> f32 {
        self.rms_norm_eps
    }
    fn vocabulary_size(&self) -> i32 {
        self.vocab_size
    }
    fn attention_bias(&self) -> bool {
        self.attention_bias
    }
    fn mlp_bias(&self) -> bool {
        self.mlp_bias
    }
    fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }
    fn attention_schedule(&self) -> &LayerSchedule<AttentionPolicy> {
        &self.attention_schedule
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

/// MLX specialization of the backend-neutral Llama block input.
pub type AttentionInput<'a, C> = eredu_architectures::llama::AttentionInput<'a, Array, C>;

/// MLX specialization of one backend-neutral Llama decoder block.
#[derive(Debug, Clone)]
pub struct TransformerBlock {
    /// Shared architecture implementation specialized to MLX operators.
    pub inner: super::backend::SharedTransformerBlock,
}

impl TransformerBlock {
    /// Creates layer zero using normalized model geometry.
    pub fn new(args: &ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        Self::new_for_layer(args, 0, stream)
    }

    pub(crate) fn new_for_layer(
        args: &ModelArgs,
        layer_index: i32,
        stream: &Stream,
    ) -> Result<Self, Exception> {
        let layer = usize::try_from(layer_index)
            .map_err(|_| Exception::custom(format!("invalid Llama layer index {layer_index}")))?;
        eredu_architectures::llama::TransformerBlock::<super::backend::MlxBackend>::new(
            args, layer, stream,
        )
        .map(|inner| Self { inner })
        .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Executes one replicated decoder block.
    pub fn forward<C>(
        &mut self,
        input: AttentionInput<'_, C>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: eredu_nn::AttentionCache<Array>,
    {
        self.inner
            .forward(input, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }

    /// Executes a block with rank-local heads and MLP intermediates.
    pub(crate) fn forward_tensor_parallel<C>(
        &mut self,
        hidden: &Array,
        mask: Option<&Array>,
        cache: Option<&mut C>,
        allow_sliding_prefill: bool,
        group: &safemlx::distributed::Group,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: eredu_nn::AttentionCache<Array>,
    {
        self.inner
            .forward_tensor_parallel(
                AttentionInput {
                    hidden,
                    mask,
                    cache,
                    allow_sliding_prefill,
                },
                group,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

fn insert_module_ref<'a>(
    map: &mut safemlx::module::ModuleParamRef<'a>,
    name: &str,
    module: &'a impl safemlx::module::ModuleParameters,
) {
    map.insert(
        name.into(),
        safemlx::nested::NestedValue::Map(module.parameters().entries),
    );
}

fn insert_module_mut<'a>(
    map: &mut safemlx::module::ModuleParamMut<'a>,
    name: &str,
    module: &'a mut impl safemlx::module::ModuleParameters,
) {
    map.insert(
        name.into(),
        safemlx::nested::NestedValue::Map(module.parameters_mut().entries),
    );
}

fn insert_trainable_ref<'a>(
    map: &mut safemlx::module::ModuleParamRef<'a>,
    name: &str,
    module: &'a impl safemlx::module::ModuleParameters,
) {
    map.insert(
        name.into(),
        safemlx::nested::NestedValue::Map(module.trainable_parameters().entries),
    );
}

fn attention_parameters(
    attention: &eredu_architectures::llama::Attention<super::backend::MlxBackend>,
) -> safemlx::module::ModuleParamRef<'_> {
    let mut map = safemlx::module::ModuleParamRef::new();
    insert_module_ref(&mut map, "q_proj", &attention.query);
    insert_module_ref(&mut map, "k_proj", &attention.key);
    insert_module_ref(&mut map, "v_proj", &attention.value);
    insert_module_ref(&mut map, "o_proj", &attention.output);
    insert_module_ref(&mut map, "rope", &attention.rotary);
    map
}

fn attention_parameters_mut(
    attention: &mut eredu_architectures::llama::Attention<super::backend::MlxBackend>,
) -> safemlx::module::ModuleParamMut<'_> {
    let mut map = safemlx::module::ModuleParamMut::new();
    insert_module_mut(&mut map, "q_proj", &mut attention.query);
    insert_module_mut(&mut map, "k_proj", &mut attention.key);
    insert_module_mut(&mut map, "v_proj", &mut attention.value);
    insert_module_mut(&mut map, "o_proj", &mut attention.output);
    insert_module_mut(&mut map, "rope", &mut attention.rotary);
    map
}

fn mlp_parameters(
    mlp: &eredu_architectures::llama::Mlp<super::backend::MlxBackend>,
) -> safemlx::module::ModuleParamRef<'_> {
    let mut map = safemlx::module::ModuleParamRef::new();
    insert_module_ref(&mut map, "gate_proj", &mlp.gate);
    insert_module_ref(&mut map, "up_proj", &mlp.up);
    insert_module_ref(&mut map, "down_proj", &mlp.down);
    map
}

fn mlp_parameters_mut(
    mlp: &mut eredu_architectures::llama::Mlp<super::backend::MlxBackend>,
) -> safemlx::module::ModuleParamMut<'_> {
    let mut map = safemlx::module::ModuleParamMut::new();
    insert_module_mut(&mut map, "gate_proj", &mut mlp.gate);
    insert_module_mut(&mut map, "up_proj", &mut mlp.up);
    insert_module_mut(&mut map, "down_proj", &mut mlp.down);
    map
}

fn block_parameters(
    block: &super::backend::SharedTransformerBlock,
) -> safemlx::module::ModuleParamRef<'_> {
    let mut map = safemlx::module::ModuleParamRef::new();
    map.insert(
        "self_attn".into(),
        safemlx::nested::NestedValue::Map(attention_parameters(&block.self_attention).entries),
    );
    map.insert(
        "mlp".into(),
        safemlx::nested::NestedValue::Map(mlp_parameters(&block.mlp).entries),
    );
    insert_module_ref(&mut map, "input_layernorm", &block.input_norm);
    insert_module_ref(
        &mut map,
        "post_attention_layernorm",
        &block.post_attention_norm,
    );
    map
}

fn block_parameters_mut(
    block: &mut super::backend::SharedTransformerBlock,
) -> safemlx::module::ModuleParamMut<'_> {
    let mut map = safemlx::module::ModuleParamMut::new();
    map.insert(
        "self_attn".into(),
        safemlx::nested::NestedValue::Map(
            attention_parameters_mut(&mut block.self_attention).entries,
        ),
    );
    map.insert(
        "mlp".into(),
        safemlx::nested::NestedValue::Map(mlp_parameters_mut(&mut block.mlp).entries),
    );
    insert_module_mut(&mut map, "input_layernorm", &mut block.input_norm);
    insert_module_mut(
        &mut map,
        "post_attention_layernorm",
        &mut block.post_attention_norm,
    );
    map
}

fn block_trainable_parameters(
    block: &super::backend::SharedTransformerBlock,
) -> safemlx::module::ModuleParamRef<'_> {
    let attention = &block.self_attention;
    let mut attention_map = safemlx::module::ModuleParamRef::new();
    insert_trainable_ref(&mut attention_map, "q_proj", &attention.query);
    insert_trainable_ref(&mut attention_map, "k_proj", &attention.key);
    insert_trainable_ref(&mut attention_map, "v_proj", &attention.value);
    insert_trainable_ref(&mut attention_map, "o_proj", &attention.output);
    insert_trainable_ref(&mut attention_map, "rope", &attention.rotary);

    let mut mlp_map = safemlx::module::ModuleParamRef::new();
    insert_trainable_ref(&mut mlp_map, "gate_proj", &block.mlp.gate);
    insert_trainable_ref(&mut mlp_map, "up_proj", &block.mlp.up);
    insert_trainable_ref(&mut mlp_map, "down_proj", &block.mlp.down);

    let mut map = safemlx::module::ModuleParamRef::new();
    map.insert(
        "self_attn".into(),
        safemlx::nested::NestedValue::Map(attention_map.entries),
    );
    map.insert(
        "mlp".into(),
        safemlx::nested::NestedValue::Map(mlp_map.entries),
    );
    insert_trainable_ref(&mut map, "input_layernorm", &block.input_norm);
    insert_trainable_ref(
        &mut map,
        "post_attention_layernorm",
        &block.post_attention_norm,
    );
    map
}

fn block_frozen_states(
    block: &super::backend::SharedTransformerBlock,
) -> impl Iterator<Item = Option<bool>> + '_ {
    [
        block.self_attention.query.all_frozen(),
        block.self_attention.key.all_frozen(),
        block.self_attention.value.all_frozen(),
        block.self_attention.output.all_frozen(),
        block.self_attention.rotary.all_frozen(),
        block.mlp.gate.all_frozen(),
        block.mlp.up.all_frozen(),
        block.mlp.down.all_frozen(),
        block.input_norm.all_frozen(),
        block.post_attention_norm.all_frozen(),
    ]
    .into_iter()
}

fn freeze_block(block: &mut super::backend::SharedTransformerBlock, mode: bool, recursive: bool) {
    let attention = &mut block.self_attention;
    let mlp = &mut block.mlp;
    if mode {
        attention.query.freeze_parameters(recursive);
        attention.key.freeze_parameters(recursive);
        attention.value.freeze_parameters(recursive);
        attention.output.freeze_parameters(recursive);
        attention.rotary.freeze_parameters(recursive);
        mlp.gate.freeze_parameters(recursive);
        mlp.up.freeze_parameters(recursive);
        mlp.down.freeze_parameters(recursive);
        block.input_norm.freeze_parameters(recursive);
        block.post_attention_norm.freeze_parameters(recursive);
    } else {
        attention.query.unfreeze_parameters(recursive);
        attention.key.unfreeze_parameters(recursive);
        attention.value.unfreeze_parameters(recursive);
        attention.output.unfreeze_parameters(recursive);
        attention.rotary.unfreeze_parameters(recursive);
        mlp.gate.unfreeze_parameters(recursive);
        mlp.up.unfreeze_parameters(recursive);
        mlp.down.unfreeze_parameters(recursive);
        block.input_norm.unfreeze_parameters(recursive);
        block.post_attention_norm.unfreeze_parameters(recursive);
    }
}

impl safemlx::module::ModuleParameters for TransformerBlock {
    fn num_parameters(&self) -> usize {
        self.parameters().flatten().len()
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        block_parameters(&self.inner)
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        block_parameters_mut(&mut self.inner)
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        block_trainable_parameters(&self.inner)
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        freeze_block(&mut self.inner, true, recursive);
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        freeze_block(&mut self.inner, false, recursive);
    }
    fn all_frozen(&self) -> Option<bool> {
        let mut states = block_frozen_states(&self.inner).flatten().peekable();
        states.peek()?;
        Some(states.all(|frozen| frozen))
    }
    fn any_frozen(&self) -> Option<bool> {
        let mut states = block_frozen_states(&self.inner).flatten().peekable();
        states.peek()?;
        Some(states.any(|frozen| frozen))
    }
}

/// Input for a resident Llama forward pass.
pub struct ModelInput<'a, C> {
    /// Token ids with shape `[batch, sequence]`.
    pub inputs: &'a Array,
    /// Optional caller-provided attention mask.
    pub mask: Option<&'a Array>,
    /// Mutable per-layer key/value cache.
    pub cache: &'a mut Vec<Option<C>>,
}

/// Resident MLX specialization of the shared Llama model.
#[derive(Debug, Clone)]
pub struct ResidentModel {
    /// Normalized model configuration and MLX weight encoding metadata.
    pub args: ModelArgs,
    /// Shared architecture specialized to native MLX modules.
    pub inner: eredu_architectures::llama::Model<super::backend::MlxBackend>,
}

impl ResidentModel {
    /// Creates an unloaded resident model.
    pub fn new(args: ModelArgs, stream: &Stream) -> Result<Self, Exception> {
        let inner =
            eredu_architectures::llama::Model::<super::backend::MlxBackend>::new(&args, stream)
                .map_err(|error| Exception::custom(error.to_string()))?;
        Ok(Self { args, inner })
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
                        i32::try_from(window.get()).expect("validated Llama window fits i32"),
                    ),
                    None => ConcatKeyValueCache::new(),
                })
            })
            .collect()
    }

    fn validate_cache<C: eredu_nn::AttentionCache<Array>>(
        &self,
        cache: &[Option<C>],
    ) -> Result<(), Exception> {
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
                i32::try_from(window.get()).expect("validated Llama window fits i32")
            });
            if eredu_nn::AttentionCache::max_size(cache) != expected {
                return Err(Exception::custom(format!("Llama cache policy mismatch at layer {layer}: expected {policy:?}, cache window is {:?}", eredu_nn::AttentionCache::max_size(cache))));
            }
        }
        Ok(())
    }

    fn forward_shared<C>(
        &mut self,
        input: ModelInput<'_, C>,
        stream: &Stream,
    ) -> Result<Array, Exception>
    where
        C: eredu_nn::AttentionCache<Array> + KeyValueCache,
    {
        self.validate_cache(input.cache)?;
        let allow_sliding_prefill = input.mask.is_none();
        let hidden = self
            .inner
            .decoder
            .embed(input.inputs, stream)
            .map_err(|error| Exception::custom(error.to_string()))?;
        let mask = match input.mask {
            Some(mask) => Some(mask.clone()),
            None => match create_attention_mask(&hidden, input.cache, Some(true), stream)? {
                Some(AttentionMask::Array(mask)) => Some(mask),
                Some(AttentionMask::Causal) => {
                    return Err(Exception::custom(
                        "Llama decoders require an explicit attention mask",
                    ))
                }
                None => None,
            },
        };
        let hidden = self
            .inner
            .decoder
            .forward_embedded(
                hidden,
                mask.as_ref(),
                allow_sliding_prefill,
                input.cache,
                stream,
            )
            .map_err(|error| Exception::custom(error.to_string()))?;
        self.inner
            .logits(&hidden, stream)
            .map_err(|error| Exception::custom(error.to_string()))
    }
}

impl<C> Module<ModelInput<'_, C>> for ResidentModel
where
    C: eredu_nn::AttentionCache<Array> + KeyValueCache,
{
    type Output = Array;
    type Error = Exception;
    fn forward(&mut self, input: ModelInput<'_, C>, stream: &Stream) -> Result<Array, Exception> {
        self.forward_shared(input, stream)
    }
    fn training_mode(&mut self, mode: bool) {
        self.inner.decoder.embeddings.training_mode(mode);
        for layer in &mut self.inner.decoder.layers {
            layer.self_attention.query.training_mode(mode);
            layer.self_attention.key.training_mode(mode);
            layer.self_attention.value.training_mode(mode);
            layer.self_attention.output.training_mode(mode);
            layer.mlp.gate.training_mode(mode);
            layer.mlp.up.training_mode(mode);
            layer.mlp.down.training_mode(mode);
            layer.input_norm.training_mode(mode);
            layer.post_attention_norm.training_mode(mode);
        }
        self.inner.decoder.norm.training_mode(mode);
        if let Some(head) = &mut self.inner.lm_head {
            head.training_mode(mode);
        }
    }
}

impl safemlx::module::ModuleParameters for ResidentModel {
    fn num_parameters(&self) -> usize {
        self.parameters().flatten().len()
    }
    fn parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        let mut model = safemlx::module::ModuleParamRef::new();
        insert_module_ref(&mut model, "embed_tokens", &self.inner.decoder.embeddings);
        let mut layers = safemlx::module::ModuleParamRef::new();
        for (index, layer) in self.inner.decoder.layers.iter().enumerate() {
            layers.insert(
                index.to_string().into(),
                safemlx::nested::NestedValue::Map(block_parameters(layer).entries),
            );
        }
        model.insert(
            "layers".into(),
            safemlx::nested::NestedValue::Map(layers.entries),
        );
        insert_module_ref(&mut model, "norm", &self.inner.decoder.norm);
        let mut root = safemlx::module::ModuleParamRef::new();
        root.insert(
            "model".into(),
            safemlx::nested::NestedValue::Map(model.entries),
        );
        if let Some(head) = &self.inner.lm_head {
            insert_module_ref(&mut root, "lm_head", head);
        }
        root
    }
    fn parameters_mut(&mut self) -> safemlx::module::ModuleParamMut<'_> {
        let mut model = safemlx::module::ModuleParamMut::new();
        insert_module_mut(
            &mut model,
            "embed_tokens",
            &mut self.inner.decoder.embeddings,
        );
        let mut layers = safemlx::module::ModuleParamMut::new();
        for (index, layer) in self.inner.decoder.layers.iter_mut().enumerate() {
            layers.insert(
                index.to_string().into(),
                safemlx::nested::NestedValue::Map(block_parameters_mut(layer).entries),
            );
        }
        model.insert(
            "layers".into(),
            safemlx::nested::NestedValue::Map(layers.entries),
        );
        insert_module_mut(&mut model, "norm", &mut self.inner.decoder.norm);
        let mut root = safemlx::module::ModuleParamMut::new();
        root.insert(
            "model".into(),
            safemlx::nested::NestedValue::Map(model.entries),
        );
        if let Some(head) = &mut self.inner.lm_head {
            insert_module_mut(&mut root, "lm_head", head);
        }
        root
    }
    fn trainable_parameters(&self) -> safemlx::module::ModuleParamRef<'_> {
        let mut model = safemlx::module::ModuleParamRef::new();
        insert_trainable_ref(&mut model, "embed_tokens", &self.inner.decoder.embeddings);
        let mut layers = safemlx::module::ModuleParamRef::new();
        for (index, layer) in self.inner.decoder.layers.iter().enumerate() {
            layers.insert(
                index.to_string().into(),
                safemlx::nested::NestedValue::Map(block_trainable_parameters(layer).entries),
            );
        }
        model.insert(
            "layers".into(),
            safemlx::nested::NestedValue::Map(layers.entries),
        );
        insert_trainable_ref(&mut model, "norm", &self.inner.decoder.norm);
        let mut root = safemlx::module::ModuleParamRef::new();
        root.insert(
            "model".into(),
            safemlx::nested::NestedValue::Map(model.entries),
        );
        if let Some(head) = &self.inner.lm_head {
            insert_trainable_ref(&mut root, "lm_head", head);
        }
        root
    }
    fn freeze_parameters(&mut self, recursive: bool) {
        self.inner.decoder.embeddings.freeze_parameters(recursive);
        for layer in &mut self.inner.decoder.layers {
            freeze_block(layer, true, recursive);
        }
        self.inner.decoder.norm.freeze_parameters(recursive);
        if let Some(head) = &mut self.inner.lm_head {
            head.freeze_parameters(recursive);
        }
    }
    fn unfreeze_parameters(&mut self, recursive: bool) {
        self.inner.decoder.embeddings.unfreeze_parameters(recursive);
        for layer in &mut self.inner.decoder.layers {
            freeze_block(layer, false, recursive);
        }
        self.inner.decoder.norm.unfreeze_parameters(recursive);
        if let Some(head) = &mut self.inner.lm_head {
            head.unfreeze_parameters(recursive);
        }
    }
    fn all_frozen(&self) -> Option<bool> {
        let mut states = std::iter::once(self.inner.decoder.embeddings.all_frozen())
            .chain(
                self.inner
                    .decoder
                    .layers
                    .iter()
                    .flat_map(block_frozen_states),
            )
            .chain(std::iter::once(self.inner.decoder.norm.all_frozen()))
            .chain(self.inner.lm_head.iter().map(ModuleParameters::all_frozen))
            .flatten()
            .peekable();
        states.peek()?;
        Some(states.all(|frozen| frozen))
    }
    fn any_frozen(&self) -> Option<bool> {
        let mut states = std::iter::once(self.inner.decoder.embeddings.any_frozen())
            .chain(self.inner.decoder.layers.iter().flat_map(|block| {
                [
                    block.self_attention.query.any_frozen(),
                    block.self_attention.key.any_frozen(),
                    block.self_attention.value.any_frozen(),
                    block.self_attention.output.any_frozen(),
                    block.self_attention.rotary.any_frozen(),
                    block.mlp.gate.any_frozen(),
                    block.mlp.up.any_frozen(),
                    block.mlp.down.any_frozen(),
                    block.input_norm.any_frozen(),
                    block.post_attention_norm.any_frozen(),
                ]
            }))
            .chain(std::iter::once(self.inner.decoder.norm.any_frozen()))
            .chain(self.inner.lm_head.iter().map(ModuleParameters::any_frozen))
            .flatten()
            .peekable();
        states.peek()?;
        Some(states.any(|frozen| frozen))
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
    C: KeyValueCache + eredu_nn::AttentionCache<Array>,
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
