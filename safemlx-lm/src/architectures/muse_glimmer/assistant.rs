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
    Array, Dtype, Stream,
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

    fn forward(
        &mut self,
        hidden: &Array,
        context: &Array,
        absolute_context_end: i32,
        window: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let batch = hidden.dim(0);
        let query_len = hidden.dim(1);
        let kv_input = concatenate_axis(&[context.clone(), hidden.clone()], 1, stream)?;
        let key_len = kv_input.dim(1);
        let mut q = self
            .q_proj
            .forward(hidden, stream)?
            .reshape(&[batch, query_len, self.heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let mut k = self
            .k_proj
            .forward(&kv_input, stream)?
            .reshape(&[batch, key_len, self.kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        let v = self
            .v_proj
            .forward(&kv_input, stream)?
            .reshape(&[batch, key_len, self.kv_heads, self.head_dim], stream)?
            .transpose_axes(&[0, 2, 1, 3], stream)?;
        q = self.q_norm.forward(&q, stream)?;
        k = self.k_norm.forward(&k, stream)?;
        q = self.rope.forward(
            nn::RopeInputBuilder::new(&q)
                .offset(absolute_context_end)
                .build()?,
            stream,
        )?;
        let context_start = absolute_context_end - context.dim(1);
        k = self.rope.forward(
            nn::RopeInputBuilder::new(&k)
                .offset(context_start)
                .build()?,
            stream,
        )?;
        let mask =
            bidirectional_block_mask(context.dim(1), query_len, absolute_context_end, window)?;
        let attended = scaled_dot_product_attention(
            q,
            k,
            v,
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
        context: &Array,
        offset: i32,
        window: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        let norm = self.input_layernorm.forward(hidden, stream)?;
        let hidden = hidden.add(
            self.self_attn
                .forward(&norm, context, offset, window, stream)?,
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

    /// Runs the anchor-plus-mask block and returns the fifteen proposal states.
    pub(crate) fn proposal_states(
        &mut self,
        noise_embeds: &Array,
        target_context: &Array,
        absolute_context_end: i32,
        stream: &Stream,
    ) -> Result<Array, Exception> {
        if noise_embeds.shape() != [1, self.config.block_size as i32, self.config.hidden_size]
            || target_context.ndim() != 3
            || target_context.dim(0) != 1
            || target_context.dim(2)
                != self.config.hidden_size * self.config.target_layer_ids.len() as i32
        {
            return Err(Exception::custom(
                "invalid Muse-Glimmer DFlash block/context geometry",
            ));
        }
        let context = self
            .encoder
            .output_norm_enc
            .forward(&self.encoder.fc.forward(target_context, stream)?, stream)?;
        let mut hidden = noise_embeds.clone();
        for layer in &mut self.layers {
            hidden = layer.forward(
                &hidden,
                &context,
                absolute_context_end,
                self.config.sliding_window,
                stream,
            )?;
        }
        let hidden = self.norm.forward(&hidden, stream)?;
        hidden.try_index_device((.., 1..self.config.block_size as i32, ..), stream)
    }
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

    use super::{translate_gguf_weight_name, DFlashConfig};

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
