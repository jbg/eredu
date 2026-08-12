//! Lossless external DFlash assistant released with Muse-Glimmer-30B.

use std::{collections::HashMap, path::Path};

use safemlx::{
    builder::Builder,
    error::Exception,
    macros::ModuleParameters,
    module::{Module, ModuleParametersExt},
    nn,
    ops::{
        concatenate_axis, indexing::TryIndexOp, GgufCheckpoint, GgufMetadataArray,
        GgufMetadataValue,
    },
    quantization::MaybeQuantized,
    transforms::async_eval_timed,
    Array, Dtype, Stream, TimedEvaluation,
};
use serde::Deserialize;
use serde_json::Value;

use crate::{
    api::ModelLoadOptions,
    error::Error,
    nn::{
        layers::silu,
        linear::unloaded_maybe_quantized_linear,
        tensor::{rope::initialize_rope, scaled_dot_product_attention},
    },
    runtime::checkpoint::{
        load::{
            gguf_metadata, gguf_quantization_configs, load_gguf_strict,
            load_safetensors_quantized_strict, load_safetensors_strict, StrictLoadConfig,
            StrictLoadReport,
        },
        quantization::WeightQuantization,
    },
};

/// Canonical DFlash checkpoint geometry.
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    /// Assistant checkpoint model type.
    pub model_type: String,
    /// Assistant hidden width.
    pub hidden_size: i32,
    /// SwiGLU intermediate width.
    pub intermediate_size: i32,
    /// Number of assistant transformer blocks.
    pub num_hidden_layers: i32,
    /// Number of query heads.
    pub num_attention_heads: i32,
    /// Number of key/value heads.
    pub num_key_value_heads: i32,
    /// Per-head width.
    pub head_dim: i32,
    /// RMS normalization epsilon.
    pub rms_norm_eps: f32,
    /// Rotary embedding frequency base.
    pub rope_theta: f32,
    /// Declared maximum position count.
    pub max_position_embeddings: i32,
    /// Accepted-context attention window.
    pub sliding_window: i32,
    /// Anchor-plus-mask proposal block size.
    pub block_size: usize,
    /// Vocabulary id used for proposal mask positions.
    pub mask_token_id: u32,
    /// Zero-based target blocks whose post-block states are consumed.
    pub target_layer_ids: Vec<usize>,
    quantization: Option<WeightQuantization>,
    quantized_weights: HashMap<String, WeightQuantization>,
}

#[derive(Deserialize)]
struct HfConfig {
    model_type: String,
    hidden_size: i32,
    intermediate_size: i32,
    num_hidden_layers: i32,
    num_attention_heads: i32,
    num_key_value_heads: i32,
    head_dim: i32,
    rms_norm_eps: f32,
    max_position_embeddings: i32,
    sliding_window: i32,
    block_size: usize,
    mask_token_id: u32,
    target_layer_ids: Vec<usize>,
    layer_types: Vec<String>,
    hidden_act: String,
    attention_dropout: f32,
    rope_parameters: HashMap<String, Value>,
}

impl DFlashConfig {
    fn from_hf(value: Value) -> Result<Self, Error> {
        let source: HfConfig = serde_json::from_value(value).map_err(|error| {
            Error::UnsupportedArchitecture(format!(
                "invalid Muse-Glimmer assistant config: {error}"
            ))
        })?;
        let rope_theta = source
            .rope_parameters
            .get("rope_theta")
            .and_then(Value::as_f64)
            .map(|value| value as f32)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(
                    "Muse-Glimmer assistant requires rope_parameters.rope_theta".into(),
                )
            })?;
        if source.model_type != "muse_glimmer_assistant"
            || source.hidden_size != 6656
            || source.intermediate_size != 19968
            || source.num_hidden_layers != 5
            || source.num_attention_heads != 32
            || source.num_key_value_heads != 8
            || source.head_dim != 128
            || source.block_size != 16
            || source.target_layer_ids != [1, 13, 25, 37, 49]
            || source.layer_types != vec!["sliding_attention"; 5]
            || source.hidden_act != "silu"
            || source.attention_dropout != 0.0
            || source.sliding_window != 2048
            || source.max_position_embeddings != 131072
            || rope_theta != 500_000.0
            || !source.rms_norm_eps.is_finite()
            || source.rms_norm_eps <= 0.0
        {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer assistant config does not match the released DFlash geometry".into(),
            ));
        }
        Ok(Self {
            model_type: source.model_type,
            hidden_size: source.hidden_size,
            intermediate_size: source.intermediate_size,
            num_hidden_layers: source.num_hidden_layers,
            num_attention_heads: source.num_attention_heads,
            num_key_value_heads: source.num_key_value_heads,
            head_dim: source.head_dim,
            rms_norm_eps: source.rms_norm_eps,
            rope_theta,
            max_position_embeddings: source.max_position_embeddings,
            sliding_window: source.sliding_window,
            block_size: source.block_size,
            mask_token_id: source.mask_token_id,
            target_layer_ids: source.target_layer_ids,
            quantization: None,
            quantized_weights: HashMap::new(),
        })
    }

    fn from_gguf(metadata: &HashMap<String, GgufMetadataValue>) -> Result<Self, Error> {
        let integer = |key: &str| {
            metadata
                .get(key)
                .and_then(GgufMetadataValue::as_i64)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DFlash GGUF requires integer metadata {key:?}"
                    ))
                })
        };
        let float = |key: &str| {
            metadata
                .get(key)
                .and_then(GgufMetadataValue::as_f32)
                .ok_or_else(|| {
                    Error::UnsupportedArchitecture(format!(
                        "DFlash GGUF requires float metadata {key:?}"
                    ))
                })
        };
        if metadata
            .get("general.architecture")
            .and_then(GgufMetadataValue::as_str)
            != Some("dflash")
        {
            return Err(Error::UnsupportedArchitecture(
                "Muse-Glimmer assistant GGUF requires general.architecture=dflash".into(),
            ));
        }
        let target_layers = match metadata.get("dflash.target_layers") {
            Some(GgufMetadataValue::Array(GgufMetadataArray::Int32(values))) => values,
            _ => {
                return Err(Error::UnsupportedArchitecture(
                    "DFlash GGUF requires Int32 dflash.target_layers".into(),
                ))
            }
        };
        // llama.cpp records one-based post-block state numbers.
        let target_layer_ids = target_layers
            .iter()
            .map(|value| {
                usize::try_from(*value)
                    .ok()
                    .and_then(|value| value.checked_sub(1))
            })
            .collect::<Option<Vec<_>>>()
            .ok_or_else(|| {
                Error::UnsupportedArchitecture("invalid DFlash target layer ids".into())
            })?;
        let config = Self {
            model_type: "muse_glimmer_assistant".into(),
            hidden_size: i32::try_from(integer("dflash.embedding_length")?).map_err(|_| {
                Error::UnsupportedArchitecture("DFlash hidden size overflow".into())
            })?,
            intermediate_size: i32::try_from(integer("dflash.feed_forward_length")?).map_err(
                |_| Error::UnsupportedArchitecture("DFlash intermediate size overflow".into()),
            )?,
            num_hidden_layers: i32::try_from(integer("dflash.block_count")?).map_err(|_| {
                Error::UnsupportedArchitecture("DFlash layer count overflow".into())
            })?,
            num_attention_heads: i32::try_from(integer("dflash.attention.head_count")?)
                .map_err(|_| Error::UnsupportedArchitecture("DFlash head count overflow".into()))?,
            num_key_value_heads: i32::try_from(integer("dflash.attention.head_count_kv")?)
                .map_err(|_| {
                    Error::UnsupportedArchitecture("DFlash KV head count overflow".into())
                })?,
            head_dim: i32::try_from(integer("dflash.attention.key_length")?)
                .map_err(|_| Error::UnsupportedArchitecture("DFlash head width overflow".into()))?,
            rms_norm_eps: float("dflash.attention.layer_norm_rms_epsilon")?,
            rope_theta: float("dflash.rope.freq_base")?,
            max_position_embeddings: i32::try_from(integer("dflash.context_length")?).map_err(
                |_| Error::UnsupportedArchitecture("DFlash context length overflow".into()),
            )?,
            sliding_window: i32::try_from(integer("dflash.attention.sliding_window")?)
                .map_err(|_| Error::UnsupportedArchitecture("DFlash window overflow".into()))?,
            block_size: usize::try_from(integer("dflash.block_size")?)
                .map_err(|_| Error::UnsupportedArchitecture("DFlash block size overflow".into()))?,
            // The released target vocabulary fixes the mask id; GGUF has no separate key.
            mask_token_id: 201818,
            target_layer_ids,
            quantization: None,
            quantized_weights: HashMap::new(),
        };
        if config.hidden_size != 6656
            || config.intermediate_size != 19968
            || config.num_hidden_layers != 5
            || config.num_attention_heads != 32
            || config.num_key_value_heads != 8
            || config.head_dim != 128
            || config.block_size != 16
            || config.target_layer_ids != [1, 13, 25, 37, 49]
            || config.sliding_window != 2048
        {
            return Err(Error::UnsupportedArchitecture(
                "DFlash GGUF geometry does not match Muse-Glimmer-30B".into(),
            ));
        }
        Ok(config)
    }

    fn quantization_for(&self, name: &str) -> Option<WeightQuantization> {
        self.quantized_weights
            .get(name)
            .copied()
            .or(self.quantization)
    }
}

#[derive(Debug, Clone, ModuleParameters)]
struct DFlashAttention {
    #[param]
    q_proj: MaybeQuantized<nn::Linear>,
    #[param]
    k_proj: MaybeQuantized<nn::Linear>,
    #[param]
    v_proj: MaybeQuantized<nn::Linear>,
    #[param]
    o_proj: MaybeQuantized<nn::Linear>,
    #[param]
    q_norm: nn::RmsNorm,
    #[param]
    k_norm: nn::RmsNorm,
    rope: crate::nn::tensor::rope::RopeVariant,
    heads: i32,
    kv_heads: i32,
    head_dim: i32,
    scale: f32,
}

#[derive(Debug, Clone)]
struct DFlashLayerContext {
    keys: Array,
    values: Array,
}

/// Per-request encoded target context and the corresponding assistant K/V.
///
/// Only committed target states are appended. Proposal-block K/V remains
/// transient, so rejecting a proposal leaves this canonical cache unchanged.
#[derive(Debug, Clone)]
pub(crate) struct DFlashContextCache {
    encoded: Array,
    layers: Vec<DFlashLayerContext>,
    start: i32,
    end: i32,
}

impl DFlashContextCache {
    fn retained_len(&self) -> i32 {
        self.end - self.start
    }

    pub(crate) fn end(&self) -> i32 {
        self.end
    }
}

impl DFlashAttention {
    fn new(config: &DFlashConfig, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("layers.{layer}.self_attn");
        let linear = |name: &str, input, output| {
            unloaded_maybe_quantized_linear(
                input,
                output,
                false,
                config.quantization_for(&format!("{prefix}.{name}.weight")),
                stream,
            )
        };
        Ok(Self {
            q_proj: linear(
                "q_proj",
                config.hidden_size,
                config.num_attention_heads * config.head_dim,
            )?,
            k_proj: linear(
                "k_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            v_proj: linear(
                "v_proj",
                config.hidden_size,
                config.num_key_value_heads * config.head_dim,
            )?,
            o_proj: linear(
                "o_proj",
                config.num_attention_heads * config.head_dim,
                config.hidden_size,
            )?,
            q_norm: nn::RmsNorm::unloaded(
                config.head_dim,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            k_norm: nn::RmsNorm::unloaded(
                config.head_dim,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            rope: initialize_rope(
                config.head_dim,
                config.rope_theta,
                false,
                &None,
                config.max_position_embeddings,
                stream,
            )?,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            scale: (config.head_dim as f32).sqrt().recip(),
        })
    }

    fn project_context(
        &mut self,
        context: &Array,
        offset: i32,
        stream: &Stream,
    ) -> Result<DFlashLayerContext, Exception> {
        let batch = context.dim(0);
        let length = context.dim(1);
        let mut keys = self
            .k_proj
            .forward(context, stream)?
            .reshape(&[batch, length, self.kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let values = self
            .v_proj
            .forward(context, stream)?
            .reshape(&[batch, length, self.kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        keys = self.k_norm.forward(&keys, stream)?;
        keys = self.rope.forward(
            nn::RopeInputBuilder::new(&keys).offset(offset).build()?,
            stream,
        )?;
        Ok(DFlashLayerContext { keys, values })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        context: &DFlashLayerContext,
        context_len: i32,
        absolute_context_end: i32,
        window: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let query_len = hidden.dim(1);
        let mut q = self
            .q_proj
            .forward(hidden, stream)?
            .reshape(&[batch, query_len, self.heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut block_keys = self
            .k_proj
            .forward(hidden, stream)?
            .reshape(&[batch, query_len, self.kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let block_values = self
            .v_proj
            .forward(hidden, stream)?
            .reshape(&[batch, query_len, self.kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        q = self.q_norm.forward(&q, stream)?;
        block_keys = self.k_norm.forward(&block_keys, stream)?;
        q = self.rope.forward(
            nn::RopeInputBuilder::new(&q)
                .offset(absolute_context_end)
                .build()?,
            stream,
        )?;
        block_keys = self.rope.forward(
            nn::RopeInputBuilder::new(&block_keys)
                .offset(absolute_context_end)
                .build()?,
            stream,
        )?;
        let keys = concatenate_axis(&[context.keys.clone(), block_keys], -2, stream)?;
        let values = concatenate_axis(&[context.values.clone(), block_values], -2, stream)?;
        let mask = bidirectional_block_mask(context_len, query_len, absolute_context_end, window)?;
        let attended = scaled_dot_product_attention(
            q,
            keys,
            values,
            Option::<crate::runtime::cache::ConcatKeyValueCache>::None,
            self.scale,
            Some(&mask),
            stream,
        )?
        .transpose_axes(&[0, 2, 1, 3], stream)?
        .reshape(&[batch, query_len, self.heads * self.head_dim], stream)?;
        self.o_proj.forward(&attended, stream)
    }
}

fn bidirectional_block_mask(
    context_len: i32,
    block_len: i32,
    context_end: i32,
    window: i32,
) -> Result<Array, Exception> {
    let key_len = context_len + block_len;
    let context_start = context_end - context_len;
    let mut values = Vec::with_capacity((block_len * key_len) as usize);
    for query in 0..block_len {
        let query_position = context_end + query;
        for key in 0..key_len {
            let key_position = context_start + key;
            let in_block = key >= context_len;
            let allowed = in_block
                || (key_position <= query_position && query_position - key_position < window);
            values.push(if allowed { 0.0 } else { f32::NEG_INFINITY });
        }
    }
    Ok(Array::from_slice(&values, &[1, 1, block_len, key_len]))
}

#[derive(Debug, Clone, ModuleParameters)]
struct DFlashMlp {
    #[param]
    gate_proj: MaybeQuantized<nn::Linear>,
    #[param]
    up_proj: MaybeQuantized<nn::Linear>,
    #[param]
    down_proj: MaybeQuantized<nn::Linear>,
}

#[derive(Debug, Clone, ModuleParameters)]
struct DFlashBlock {
    #[param]
    input_layernorm: nn::RmsNorm,
    #[param]
    self_attn: DFlashAttention,
    #[param]
    post_attention_layernorm: nn::RmsNorm,
    #[param]
    mlp: DFlashMlp,
}

impl DFlashBlock {
    fn new(config: &DFlashConfig, layer: usize, stream: &Stream) -> Result<Self, Exception> {
        let prefix = format!("layers.{layer}.mlp");
        let linear = |name: &str, input, output| {
            unloaded_maybe_quantized_linear(
                input,
                output,
                false,
                config.quantization_for(&format!("{prefix}.{name}.weight")),
                stream,
            )
        };
        Ok(Self {
            input_layernorm: nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            self_attn: DFlashAttention::new(config, layer, stream)?,
            post_attention_layernorm: nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
            mlp: DFlashMlp {
                gate_proj: linear("gate_proj", config.hidden_size, config.intermediate_size)?,
                up_proj: linear("up_proj", config.hidden_size, config.intermediate_size)?,
                down_proj: linear("down_proj", config.intermediate_size, config.hidden_size)?,
            },
        })
    }

    fn forward(
        &mut self,
        hidden: &Array,
        context: &DFlashLayerContext,
        context_len: i32,
        offset: i32,
        window: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let norm = self.input_layernorm.forward(hidden, stream)?;
        let hidden = hidden.add(
            self.self_attn
                .forward(&norm, context, context_len, offset, window, stream)?,
            stream,
        )?;
        let norm = self.post_attention_layernorm.forward(&hidden, stream)?;
        let gate = silu(self.mlp.gate_proj.forward(&norm, stream)?, stream)?;
        let mlp = self.mlp.down_proj.forward(
            &gate.multiply(self.mlp.up_proj.forward(&norm, stream)?, stream)?,
            stream,
        )?;
        hidden.add(mlp, stream)
    }
}

/// Fully resident DFlash assistant body. Target embedding/head snapshots are
/// deliberately supplied by the target backend and are not checkpoint fields.
#[derive(Debug, Clone, ModuleParameters)]
pub struct MuseGlimmerDFlash {
    #[param]
    encoder: DFlashEncoder,
    #[param]
    layers: Vec<DFlashBlock>,
    #[param]
    norm: nn::RmsNorm,
    /// Validated assistant checkpoint geometry.
    pub config: DFlashConfig,
}

#[derive(Debug, Clone, ModuleParameters)]
struct DFlashEncoder {
    #[param]
    fc: MaybeQuantized<nn::Linear>,
    #[param]
    output_norm_enc: nn::RmsNorm,
}

impl MuseGlimmerDFlash {
    fn new(config: DFlashConfig, stream: &Stream) -> Result<Self, Exception> {
        let encoder = DFlashEncoder {
            fc: unloaded_maybe_quantized_linear(
                config.hidden_size * config.target_layer_ids.len() as i32,
                config.hidden_size,
                false,
                config.quantization_for("encoder.fc.weight"),
                stream,
            )?,
            output_norm_enc: nn::RmsNorm::unloaded(
                config.hidden_size,
                config.rms_norm_eps,
                Dtype::Float32,
                stream,
            )?,
        };
        let layers = (0..config.num_hidden_layers as usize)
            .map(|layer| DFlashBlock::new(&config, layer, stream))
            .collect::<Result<Vec<_>, _>>()?;
        let norm = nn::RmsNorm::unloaded(
            config.hidden_size,
            config.rms_norm_eps,
            Dtype::Float32,
            stream,
        )?;
        Ok(Self {
            encoder,
            layers,
            norm,
            config,
        })
    }

    /// Incrementally encodes newly committed target states and appends their
    /// per-layer K/V projections to the request's canonical draft cache.
    pub(crate) fn update_context_cache(
        &mut self,
        previous: Option<DFlashContextCache>,
        pending_target_states: Option<&Array>,
        absolute_context_end: i32,
        component_timing: bool,
        stream: &Stream,
    ) -> Result<(DFlashContextCache, Option<TimedEvaluation>), Exception> {
        let pending_len = pending_target_states.map_or(0, |states| states.dim(1));
        let pending_start = context_append_start(
            previous.as_ref().map(DFlashContextCache::end),
            pending_len,
            absolute_context_end,
        )?;
        if pending_len == 0 {
            return previous.map(|cache| (cache, None)).ok_or_else(|| {
                Exception::custom("Muse-Glimmer DFlash cache cannot start from empty context")
            });
        }
        let pending = pending_target_states.expect("non-empty pending context");
        if pending.ndim() != 3
            || pending.dim(0) != 1
            || pending.dim(2) != self.config.hidden_size * self.config.target_layer_ids.len() as i32
        {
            return Err(Exception::custom(
                "invalid Muse-Glimmer DFlash target context geometry",
            ));
        }
        let encoded = self
            .encoder
            .output_norm_enc
            .forward(&self.encoder.fc.forward(pending, stream)?, stream)?;
        let layer_chunks = self
            .layers
            .iter_mut()
            .map(|layer| {
                layer
                    .self_attn
                    .project_context(&encoded, pending_start, stream)
            })
            .collect::<Result<Vec<_>, _>>()?;

        let window = self.config.sliding_window;
        let (encoded, layers) = if let Some(previous) = previous {
            if previous.layers.len() != layer_chunks.len()
                || previous.encoded.dim(1) != previous.retained_len()
            {
                return Err(Exception::custom(
                    "invalid Muse-Glimmer DFlash cached context geometry",
                ));
            }
            let encoded = append_encoded_tail(previous.encoded, encoded, window, stream)?;
            let layers = previous
                .layers
                .into_iter()
                .zip(layer_chunks)
                .map(|(previous, chunk)| append_layer_tail(previous, chunk, window, stream))
                .collect::<Result<Vec<_>, _>>()?;
            (encoded, layers)
        } else {
            let encoded = retain_encoded_tail(encoded, window, stream)?;
            let layers = layer_chunks
                .into_iter()
                .map(|layer| retain_layer_tail(layer, window, stream))
                .collect::<Result<Vec<_>, _>>()?;
            (encoded, layers)
        };
        let retained = encoded.dim(1);
        let cache = DFlashContextCache {
            encoded,
            layers,
            start: absolute_context_end - retained,
            end: absolute_context_end,
        };
        let timing = if component_timing {
            Some(async_eval_timed(
                std::iter::once(&cache.encoded).chain(
                    cache
                        .layers
                        .iter()
                        .flat_map(|layer| [&layer.keys, &layer.values]),
                ),
                stream,
            )?)
        } else {
            None
        };
        Ok((cache, timing))
    }

    /// Runs an anchor-plus-mask block up to the released maximum width.
    pub(crate) fn proposal_states(
        &mut self,
        noise_embeds: &Array,
        context: &DFlashContextCache,
        absolute_context_end: i32,
        component_timing: bool,
        stream: &Stream,
    ) -> Result<(Array, Option<TimedEvaluation>), Exception> {
        if noise_embeds.ndim() != 3 {
            return Err(Exception::custom(
                "invalid Muse-Glimmer DFlash block/context geometry",
            ));
        }
        let runtime_block_size = noise_embeds.dim(1);
        if noise_embeds.dim(0) != 1
            || runtime_block_size < 2
            || runtime_block_size > self.config.block_size as i32
            || noise_embeds.dim(2) != self.config.hidden_size
            || context.end != absolute_context_end
            || context.encoded.ndim() != 3
            || context.encoded.dim(0) != 1
            || context.encoded.dim(2) != self.config.hidden_size
            || context.encoded.dim(1) != context.retained_len()
            || context.layers.len() != self.layers.len()
        {
            return Err(Exception::custom(
                "invalid Muse-Glimmer DFlash block/context geometry",
            ));
        }
        let context_len = context.retained_len();
        let mut hidden = noise_embeds.clone();
        for (layer, layer_context) in self.layers.iter_mut().zip(&context.layers) {
            if layer_context.keys.dim(-2) != context_len
                || layer_context.values.dim(-2) != context_len
            {
                return Err(Exception::custom(
                    "invalid Muse-Glimmer DFlash layer-cache geometry",
                ));
            }
            hidden = layer.forward(
                &hidden,
                layer_context,
                context_len,
                absolute_context_end,
                self.config.sliding_window,
                stream,
            )?;
        }
        let hidden = self.norm.forward(&hidden, stream)?;
        let states = hidden.try_index_device((.., 1..runtime_block_size, ..), stream)?;
        let timing = component_timing
            .then(|| async_eval_timed([&states], stream))
            .transpose()?;
        Ok((states, timing))
    }
}

fn context_append_start(
    previous_end: Option<i32>,
    pending_len: i32,
    absolute_end: i32,
) -> Result<i32, Exception> {
    if pending_len < 0 || absolute_end < pending_len {
        return Err(Exception::custom(
            "invalid Muse-Glimmer DFlash context range",
        ));
    }
    let start = absolute_end - pending_len;
    if let Some(previous_end) = previous_end {
        if previous_end != start {
            return Err(Exception::custom(format!(
                "Muse-Glimmer DFlash context/cache frontier mismatch: cached through {previous_end}, pending starts at {start}"
            )));
        }
    }
    Ok(start)
}

fn retain_encoded_tail(encoded: Array, window: i32, stream: &Stream) -> Result<Array, Exception> {
    let start = (encoded.dim(1) - window).max(0);
    encoded.try_index_device((.., start.., ..), stream)
}

fn retain_layer_tail(
    layer: DFlashLayerContext,
    window: i32,
    stream: &Stream,
) -> Result<DFlashLayerContext, Exception> {
    let start = (layer.keys.dim(-2) - window).max(0);
    Ok(DFlashLayerContext {
        keys: layer.keys.try_index_device((.., .., start.., ..), stream)?,
        values: layer
            .values
            .try_index_device((.., .., start.., ..), stream)?,
    })
}

fn append_encoded_tail(
    previous: Array,
    pending: Array,
    window: i32,
    stream: &Stream,
) -> Result<Array, Exception> {
    retain_encoded_tail(
        concatenate_axis(&[previous, pending], 1, stream)?,
        window,
        stream,
    )
}

fn append_layer_tail(
    previous: DFlashLayerContext,
    pending: DFlashLayerContext,
    window: i32,
    stream: &Stream,
) -> Result<DFlashLayerContext, Exception> {
    retain_layer_tail(
        DFlashLayerContext {
            keys: concatenate_axis(&[previous.keys, pending.keys], -2, stream)?,
            values: concatenate_axis(&[previous.values, pending.values], -2, stream)?,
        },
        window,
        stream,
    )
}

/// Loads either the official safetensors assistant or `dflash-kquant.gguf`.
pub(crate) fn load_with_options(
    source: &Path,
    options: ModelLoadOptions,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<MuseGlimmerDFlash, Error> {
    if !options.weight_residency.is_fully_resident() {
        return Err(Error::UnsupportedArchitecture(
            "Muse-Glimmer DFlash currently requires fully resident assistant weights".into(),
        ));
    }
    if source
        .extension()
        .and_then(|value| value.to_str())
        .is_some_and(|value| value.eq_ignore_ascii_case("gguf"))
    {
        let checkpoint = GgufCheckpoint::open(source)?;
        let metadata = gguf_metadata(&checkpoint);
        let mut config = DFlashConfig::from_gguf(&metadata)?;
        config.quantized_weights =
            gguf_quantization_configs(&checkpoint, translate_gguf_weight_name)?;
        if let Some(requested) = options.quantization {
            crate::api::validate_gguf_quantization_source(&checkpoint, &metadata, Some(requested))?;
        }
        let mut model = MuseGlimmerDFlash::new(config, stream)?;
        let load_config = StrictLoadConfig::default();
        let mut report = StrictLoadReport::default();
        load_gguf_strict(
            &mut model,
            &checkpoint,
            None,
            &load_config,
            &mut report,
            |name, value| Ok((translate_gguf_weight_name(&name), value)),
        )?;
        report.finish(&model, &load_config)?;
        model.copy_to_stream(stream)?;
        return Ok(model);
    }
    let value: Value = serde_json::from_reader(std::fs::File::open(source.join("config.json"))?)?;
    let mut config = DFlashConfig::from_hf(value)?;
    let quantize = if let Some(requested) = options.quantization {
        config.quantization = Some(requested);
        true
    } else {
        false
    };
    let mut model = MuseGlimmerDFlash::new(config, stream)?;
    let load_config = StrictLoadConfig::default();
    let mut report = StrictLoadReport::default();
    let path = source.join("model.safetensors");
    if quantize {
        load_safetensors_quantized_strict(
            &mut model,
            path,
            weights_stream,
            stream,
            options.quantization.expect("checked"),
            &load_config,
            &mut report,
        )?;
    } else {
        load_safetensors_strict(&mut model, path, weights_stream, &load_config, &mut report)?;
    }
    report.finish(&model, &load_config)?;
    model.copy_to_stream(stream)?;
    Ok(model)
}

pub(crate) fn translate_gguf_weight_name(name: &str) -> String {
    match name {
        "fc.weight" => return "encoder.fc.weight".into(),
        "enc.output_norm.weight" => return "encoder.output_norm_enc.weight".into(),
        "output_norm.weight" => return "norm.weight".into(),
        _ => {}
    }
    let Some(rest) = name.strip_prefix("blk.") else {
        return name.into();
    };
    let Some((layer, parameter)) = rest.split_once('.') else {
        return name.into();
    };
    for (source, target) in [
        ("attn_norm", "input_layernorm"),
        ("ffn_norm", "post_attention_layernorm"),
        ("attn_q_norm", "self_attn.q_norm"),
        ("attn_k_norm", "self_attn.k_norm"),
        ("attn_q", "self_attn.q_proj"),
        ("attn_k", "self_attn.k_proj"),
        ("attn_v", "self_attn.v_proj"),
        ("attn_output", "self_attn.o_proj"),
        ("ffn_gate", "mlp.gate_proj"),
        ("ffn_up", "mlp.up_proj"),
        ("ffn_down", "mlp.down_proj"),
    ] {
        if parameter == source || parameter.starts_with(&format!("{source}.")) {
            return format!("layers.{layer}.{}", parameter.replacen(source, target, 1));
        }
    }
    name.into()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{context_append_start, translate_gguf_weight_name, DFlashConfig};

    #[test]
    fn committed_context_updates_must_continue_the_cached_frontier() {
        assert_eq!(context_append_start(None, 75, 75).unwrap(), 0);
        assert_eq!(context_append_start(None, 2048, 4096).unwrap(), 2048);
        assert_eq!(context_append_start(Some(4096), 3, 4099).unwrap(), 4096);

        assert!(context_append_start(Some(4096), 2, 4099).is_err());
        assert!(context_append_start(Some(4100), 3, 4099).is_err());
    }

    #[test]
    fn translates_official_dflash_names() {
        assert_eq!(translate_gguf_weight_name("fc.weight"), "encoder.fc.weight");
        assert_eq!(
            translate_gguf_weight_name("blk.4.attn_output.weight"),
            "layers.4.self_attn.o_proj.weight"
        );
        assert_eq!(
            translate_gguf_weight_name("blk.2.ffn_norm.weight"),
            "layers.2.post_attention_layernorm.weight"
        );
    }

    #[test]
    fn accepts_only_the_released_dflash_geometry() {
        let released = json!({
            "model_type": "muse_glimmer_assistant",
            "hidden_size": 6656,
            "intermediate_size": 19968,
            "num_hidden_layers": 5,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 131072,
            "sliding_window": 2048,
            "block_size": 16,
            "mask_token_id": 201818,
            "target_layer_ids": [1, 13, 25, 37, 49],
            "layer_types": ["sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
            "hidden_act": "silu",
            "attention_dropout": 0.0,
            "rope_parameters": {"rope_theta": 500000.0}
        });
        let parsed = DFlashConfig::from_hf(released.clone()).unwrap();
        assert_eq!(parsed.target_layer_ids, [1, 13, 25, 37, 49]);
        assert_eq!(parsed.block_size - 1, 15);

        let mut wrong = released;
        wrong["target_layer_ids"][2] = json!(24);
        assert!(DFlashConfig::from_hf(wrong).is_err());
    }
}
