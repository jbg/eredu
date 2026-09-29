//! Distinct normalization of the released qwen4_exp/qwen4_exp_text contract.
use crate::qwen::vision::{VisionConfig, VisionConfigSource};
use eredu_core::LayerSchedule;
use eredu_nn::OutputGateActivation;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};

/// Full-attention checkpoint labels denote QSA within this family.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Recurrent gated delta mixer.
    Recurrent,
    /// Indexed sparse attention over original K/V history.
    Indexed,
}
/// Multi-stream residual geometry.
#[derive(Debug, Clone, PartialEq)]
pub struct ResidualGeometry {
    /// Parallel residual streams.
    pub streams: i32,
    /// Rank of the residual input mixing projection.
    pub rank: i32,
}
/// Attention, rotary and QSA geometry.
#[derive(Debug, Clone)]
pub struct AttentionGeometry {
    /// Ordinary query heads.
    pub heads: i32,
    /// Ordinary K/V heads, replicated when TP exceeds their count.
    pub kv_heads: i32,
    /// Attention head width.
    pub head_dim: i32,
    /// Shared ordinary/indexer rotary width.
    pub rotary_dim: i32,
    /// Complete architecture-owned rotary declaration.
    pub rope: HashMap<String, Value>,
    /// Validated ordinary/indexer rotary operator with released rounding boundaries.
    pub rotary: eredu_nn::RotarySpec,
    /// QSA query heads.
    pub index_heads: i32,
    /// QSA key head count (one in the released equation).
    pub index_kv_heads: i32,
    /// Indexer head width.
    pub index_head_dim: i32,
    /// Token budget drawn from complete blocks.
    pub budget: i32,
    /// Visible tokens per complete micro-block.
    pub ratio: i32,
    /// Whether ordinary Q/K/V projections include bias.
    pub bias: bool,
}
/// Reusable gated-delta mixer geometry and explicit output activation.
#[derive(Debug, Clone, PartialEq)]
pub struct RecurrentGeometry {
    /// Query/key head count.
    pub key_heads: i32,
    /// Value/output head count.
    pub value_heads: i32,
    /// Key head width.
    pub key_dim: i32,
    /// Value head width.
    pub value_dim: i32,
    /// Causal convolution tap count, dilation one.
    pub kernel: i32,
    /// Released Flash-Next selects sigmoid.
    pub gate: OutputGateActivation,
}
/// Independently addressable routed experts and shared feed-forward branch.
#[derive(Debug, Clone, PartialEq)]
pub struct ExpertGeometry {
    /// Global expert count.
    pub count: i32,
    /// Selected experts per token.
    pub selected: i32,
    /// Routed intermediate width.
    pub intermediate: i32,
    /// Always-on shared intermediate width.
    pub shared_intermediate: i32,
    /// Normalize selected routing probabilities.
    pub renormalize: bool,
}
/// Source facts omitted by GGUF must not be reconstructed from model defaults.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NGramSourceLayout {
    /// Original sharded SafeTensors declaration, including hash-generation provenance.
    Safetensors {
        /// Lower bound used when generating prime head vocabularies.
        vocabulary_base: u64,
        /// Padding alignment of the concatenated table.
        vocabulary_alignment: u64,
        /// Declared physical embedding shard count.
        shards: usize,
        /// Exact initialization seed; execution consumes literal checkpoint controls.
        seed: u64,
    },
    /// The converter retains one physical table and literal hash metadata. It omits
    /// the original seed, shard split, prime-generation base and padding alignment.
    Gguf {
        /// Exact physical row count, including stored padding.
        rows: u64,
    },
}
/// N-gram injection and compact table-source geometry.
#[derive(Debug, Clone)]
pub struct NGramGeometry {
    /// Zero-based injection layers, normalized from one-based checkpoint IDs.
    pub layers: Vec<usize>,
    /// Maximum token order and convolution dilation.
    pub order: i32,
    /// Hash heads per order.
    pub heads: i32,
    /// Exact source-specific table declaration.
    pub source: NGramSourceLayout,
    /// Concatenated n-gram embedding width.
    pub embedding_dim: i32,
    /// Dilated convolution tap count.
    pub kernel: i32,
}
/// Embedded prediction schedule; weights and state have independent ownership.
#[derive(Debug, Clone)]
pub struct PredictionGeometry {
    /// Prediction layers in their declared order.
    pub layers: LayerSchedule<LayerKind>,
    /// Prediction rotary base.
    pub rope_theta: f32,
}
/// Released visual token identities retained for shared media ingress.
#[derive(Debug, Clone)]
pub struct MediaTokens {
    /// Image placeholder token.
    pub image: u32,
    /// Video placeholder token.
    pub video: u32,
    /// Visual segment start.
    pub start: u32,
    /// Visual segment end.
    pub end: u32,
}
/// Complete family configuration, independent of Qwen3.5 text normalization.
#[derive(Debug, Clone)]
pub struct Config {
    /// Token vocabulary count.
    pub vocabulary: i32,
    /// Base hidden width before residual expansion.
    pub hidden_size: i32,
    /// Maximum declared sequence positions.
    pub max_positions: i32,
    /// RMS epsilon for family equations.
    pub norm_epsilon: f32,
    /// EOS IDs; the first is the n-gram reset/padding token.
    pub eos: Vec<u32>,
    /// Whether target output and token embeddings share a parameter.
    pub tied_embeddings: bool,
    /// Authoritative recurrent/indexed schedule.
    pub layers: LayerSchedule<LayerKind>,
    /// Residual stream geometry.
    pub residual: ResidualGeometry,
    /// Attention/indexer geometry.
    pub attention: AttentionGeometry,
    /// Recurrent mixer geometry.
    pub recurrent: RecurrentGeometry,
    /// Routed/shared feed-forward geometry.
    pub experts: ExpertGeometry,
    /// Explicit injection and table geometry.
    pub ngram: NGramGeometry,
    /// Embedded prediction declaration, when present.
    pub prediction: Option<PredictionGeometry>,
    /// Shared vision encoder normalization.
    pub vision: Option<VisionConfig>,
    /// Media placeholder policy accompanying vision.
    pub media: Option<MediaTokens>,
}
/// Invalid family identity, required field or incompatible equation geometry.
#[derive(Debug, thiserror::Error)]
#[error("invalid qwen4_exp configuration: {0}")]
pub struct ConfigError(pub String);
fn invalid(message: impl Into<String>) -> ConfigError {
    ConfigError(message.into())
}
fn integer(value: &Value, name: &str) -> Result<i32, ConfigError> {
    value
        .get(name)
        .and_then(Value::as_i64)
        .and_then(|v| i32::try_from(v).ok())
        .filter(|v| *v > 0)
        .ok_or_else(|| invalid(format!("{name} must be a positive i32 integer")))
}
fn unsigned(value: &Value, name: &str, default: Option<u64>) -> Result<u64, ConfigError> {
    match value.get(name) {
        None => default.ok_or_else(|| invalid(format!("missing {name}"))),
        Some(value) => value
            .as_u64()
            .ok_or_else(|| invalid(format!("{name} must be an exact unsigned integer"))),
    }
}
fn boolean(value: &Value, name: &str, default: bool) -> Result<bool, ConfigError> {
    value.get(name).map_or(Ok(default), |v| {
        v.as_bool()
            .ok_or_else(|| invalid(format!("{name} must be boolean")))
    })
}
fn schedule(
    value: Option<&Value>,
    count: usize,
    interval: usize,
) -> Result<LayerSchedule<LayerKind>, ConfigError> {
    let layers = if let Some(value) = value {
        let values = value
            .as_array()
            .filter(|values| values.len() == count)
            .ok_or_else(|| invalid("layer_types must exactly cover the declared layer count"))?;
        values
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(LayerKind::Recurrent),
                Some("full_attention" | "indexed_attention") => Ok(LayerKind::Indexed),
                _ => Err(invalid("unknown qwen4_exp layer type")),
            })
            .collect::<Result<Vec<_>, _>>()?
    } else {
        if interval == 0 {
            return Err(invalid("full_attention_interval must be positive"));
        }
        (0..count)
            .map(|layer| {
                if (layer + 1) % interval == 0 {
                    LayerKind::Indexed
                } else {
                    LayerKind::Recurrent
                }
            })
            .collect()
    };
    LayerSchedule::new(count, layers).map_err(|e| invalid(e.to_string()))
}
impl Config {
    /// Parses only this family's text or conditional-generation configuration.
    pub fn from_json(root: &Value) -> Result<Self, ConfigError> {
        Self::from_document(root, None)
    }
    // Both containers normalize through the same equation checks. GGUF supplies
    // only its actual source geometry, never synthetic SafeTensors provenance.
    fn from_document(root: &Value, source: Option<NGramSourceLayout>) -> Result<Self, ConfigError> {
        let text = match root.get("model_type").and_then(Value::as_str) {
            Some("qwen4_exp") => root
                .get("text_config")
                .ok_or_else(|| invalid("conditional model omitted text_config"))?,
            Some("qwen4_exp_text") => root,
            _ => return Err(invalid("expected qwen4_exp or qwen4_exp_text")),
        };
        if text.get("model_type").and_then(Value::as_str) != Some("qwen4_exp_text") {
            return Err(invalid("text_config has a different family identity"));
        }
        let vocabulary = integer(text, "vocab_size")?;
        let hidden_size = integer(text, "hidden_size")?;
        let count = integer(text, "num_hidden_layers")? as usize;
        let layers = schedule(
            text.get("layer_types"),
            count,
            usize::try_from(unsigned(text, "full_attention_interval", Some(4))?)
                .map_err(|_| invalid("attention interval exceeds host range"))?,
        )?;
        let epsilon = text
            .get("rms_norm_eps")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid("missing RMS epsilon"))? as f32;
        if !epsilon.is_finite() || epsilon <= 0. {
            return Err(invalid("RMS epsilon must be positive and finite"));
        }
        let eos = match text.get("eos_token_id") {
            Some(Value::Array(ids)) => ids.iter().map(|v| v.as_u64()).collect::<Option<Vec<_>>>(),
            Some(id) => id.as_u64().map(|id| vec![id]),
            None => None,
        }
        .filter(|ids| !ids.is_empty() && ids.iter().all(|id| *id < vocabulary as u64))
        .ok_or_else(|| invalid("invalid EOS token IDs"))?
        .into_iter()
        .map(|id| id as u32)
        .collect();
        if text.get("hidden_act").and_then(Value::as_str) != Some("silu") {
            return Err(invalid("released feed-forward activation requires silu"));
        }
        let gate = match text
            .get("output_gate_type")
            .filter(|v| !v.is_null())
            .map(|v| {
                v.as_str()
                    .ok_or_else(|| invalid("output_gate_type must be a string"))
            })
            .transpose()?
            .unwrap_or("silu")
        {
            "silu" => OutputGateActivation::Silu,
            "sigmoid" => OutputGateActivation::Sigmoid,
            _ => return Err(invalid("unsupported recurrent output gate")),
        };
        let residual = ResidualGeometry {
            streams: integer(text, "hc_count")?,
            rank: integer(text, "hc_lowrank")?,
        };
        if residual.streams <= 1 || hidden_size.checked_mul(residual.streams).is_none() {
            return Err(invalid("invalid residual stream width"));
        }
        let rope = text
            .get("rope_parameters")
            .and_then(Value::as_object)
            .ok_or_else(|| invalid("missing rope_parameters"))?;
        let head_dim = integer(text, "head_dim")?;
        let factor = match rope.get("partial_rotary_factor") {
            None => 1.,
            Some(value) => value
                .as_f64()
                .ok_or_else(|| invalid("partial_rotary_factor must be numeric"))?,
        };
        let rotary_exact = head_dim as f64 * factor;
        let rotary = rotary_exact as i32;
        if !factor.is_finite()
            || !(0.0..=1.0).contains(&factor)
            || rotary_exact != rotary as f64
            || rotary <= 0
            || rotary % 2 != 0
        {
            return Err(invalid("invalid partial rotary width"));
        }
        let theta = rope
            .get("rope_theta")
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| invalid("rotary base must be positive and finite"))?
            as f32;
        if !theta.is_finite() || theta <= 0. {
            return Err(invalid("rotary base exceeds its floating representation"));
        }
        let algorithm_values = rope
            .iter()
            .filter(|(key, _)| key.as_str() != "mrope_section")
            .map(|(key, value)| {
                serde_json::from_value::<crate::rotary::RopeValue>(value.clone())
                    .map(|value| (key.clone(), value))
                    .map_err(|e| invalid(e.to_string()))
            })
            .collect::<Result<HashMap<_, _>, _>>()?;
        let algorithm =
            crate::rotary::normalize_algorithm(Some(&algorithm_values)).map_err(invalid)?;
        if let Some(sections) = rope.get("mrope_section") {
            let sections = sections
                .as_array()
                .filter(|s| s.len() == 3)
                .ok_or_else(|| invalid("media rotary sections must have three dimensions"))?;
            let total = sections
                .iter()
                .try_fold(0u64, |total, value| {
                    total.checked_add(value.as_u64().filter(|v| *v > 0)?)
                })
                .ok_or_else(|| invalid("media rotary sections must be positive integers"))?;
            if total.checked_mul(2) != Some(rotary as u64) {
                return Err(invalid("media rotary sections must cover the rotary width"));
            }
        }
        if rope
            .get("mrope_interleaved")
            .is_some_and(|v| v.as_bool().is_none())
        {
            return Err(invalid("mrope_interleaved must be boolean"));
        }
        let attention = AttentionGeometry {
            heads: integer(text, "num_attention_heads")?,
            kv_heads: integer(text, "num_key_value_heads")?,
            head_dim,
            rotary_dim: rotary,
            rope: rope.clone().into_iter().collect(),
            rotary: eredu_nn::RotarySpec {
                dimensions: rotary,
                base: theta,
                traditional: false,
                algorithm,
                arithmetic: eredu_nn::RotaryArithmetic::InputProducts,
            },
            index_heads: integer(text, "indexer_n_heads")?,
            index_kv_heads: integer(text, "indexer_kv_heads")?,
            index_head_dim: integer(text, "indexer_head_dim")?,
            budget: integer(text, "indexer_budget")?,
            ratio: integer(text, "indexer_compress_ratio")?,
            bias: boolean(text, "attention_bias", false)?,
        };
        if attention.heads % attention.kv_heads != 0
            || attention.index_kv_heads != 1
            || attention.rotary_dim > attention.index_head_dim
            || attention.budget % attention.ratio != 0
        {
            return Err(invalid("invalid GQA or QSA geometry"));
        }
        let recurrent = RecurrentGeometry {
            key_heads: integer(text, "linear_num_key_heads")?,
            value_heads: integer(text, "linear_num_value_heads")?,
            key_dim: integer(text, "linear_key_head_dim")?,
            value_dim: integer(text, "linear_value_head_dim")?,
            kernel: integer(text, "linear_conv_kernel_dim")?,
            gate,
        };
        if recurrent.value_heads % recurrent.key_heads != 0 {
            return Err(invalid("recurrent value heads must group over key heads"));
        }
        let experts = ExpertGeometry {
            count: integer(text, "num_experts")?,
            selected: integer(text, "num_experts_per_tok")?,
            intermediate: integer(text, "moe_intermediate_size")?,
            shared_intermediate: integer(text, "shared_expert_intermediate_size")?,
            renormalize: boolean(text, "norm_topk_prob", true)?,
        };
        if experts.selected > experts.count {
            return Err(invalid("expert selection exceeds bank cardinality"));
        }
        let injection = text
            .get("ple_layer_ids")
            .and_then(Value::as_array)
            .ok_or_else(|| invalid("missing ple_layer_ids"))?;
        let mut injection_layers = BTreeSet::new();
        for layer in injection {
            let layer = layer
                .as_u64()
                .filter(|v| *v >= 1 && *v <= count as u64)
                .ok_or_else(|| {
                    invalid("PLE layer IDs are one-based and must be inside the decoder")
                })?;
            if layers.get(layer as usize - 1) != Some(&LayerKind::Recurrent) {
                return Err(invalid("PLE injection requires a recurrent layer"));
            }
            if !injection_layers.insert(layer as usize - 1) {
                return Err(invalid("duplicate PLE layer ID"));
            }
        }
        let ngram = NGramGeometry {
            layers: injection_layers.into_iter().collect(),
            order: integer(text, "ngram_size")?,
            heads: integer(text, "heads_per_ngram")?,
            source: match source {
                Some(source) => source,
                None => NGramSourceLayout::Safetensors {
                    vocabulary_base: unsigned(text, "ngram_vocab_size_base", None)?,
                    vocabulary_alignment: unsigned(
                        text,
                        "make_ngram_vocab_size_divisible_by",
                        Some(128),
                    )?,
                    shards: usize::try_from(unsigned(text, "split_ngram_parts", Some(512))?)
                        .map_err(|_| invalid("too many table shards"))?,
                    seed: unsigned(text, "seed", Some(1234))?,
                },
            },
            embedding_dim: integer(text, "ple_embed_dim").or_else(|_| {
                if text.get("ple_embed_dim").is_none_or(Value::is_null) {
                    Ok(hidden_size)
                } else {
                    Err(invalid("invalid ple_embed_dim"))
                }
            })?,
            kernel: integer(text, "ple_conv_kernel_size")?,
        };
        let hash_heads = ngram
            .order
            .checked_sub(1)
            .and_then(|v| v.checked_mul(ngram.heads))
            .filter(|v| *v > 0)
            .ok_or_else(|| invalid("invalid n-gram heads"))?;
        if ngram.embedding_dim % hash_heads != 0
            || match ngram.source {
                NGramSourceLayout::Safetensors {
                    shards,
                    vocabulary_alignment,
                    vocabulary_base,
                    ..
                } => shards == 0 || vocabulary_alignment == 0 || vocabulary_base < 2,
                NGramSourceLayout::Gguf { rows } => rows == 0,
            }
            || (ngram.kernel - 1).checked_mul(ngram.order).is_none()
        {
            return Err(invalid("invalid n-gram table or convolution geometry"));
        }
        let prediction = match text.get("mtp") {
            None | Some(Value::Null) => {
                if text
                    .get("mtp_num_hidden_layers")
                    .is_some_and(|v| v.as_u64() != Some(0))
                {
                    return Err(invalid("prediction depth requires mtp configuration"));
                }
                None
            }
            Some(mtp) => {
                let depth = integer(mtp, "num_hidden_layers")? as usize;
                if !boolean(mtp, "hybrid", true)?
                    || boolean(text, "mtp_use_dedicated_embeddings", false)?
                    || mtp
                        .get("mtp_use_hidden_state_from_layer")
                        .is_some_and(|v| !v.is_null())
                {
                    return Err(invalid("prediction must fuse retained residual streams and shared embedding/output parameters"));
                }
                if text.get("mtp_num_hidden_layers").is_some()
                    && integer(text, "mtp_num_hidden_layers")? as usize != depth
                {
                    return Err(invalid("prediction depth declarations differ"));
                }
                let theta = mtp
                    .get("rope_theta")
                    .and_then(Value::as_f64)
                    .filter(|v| v.is_finite() && *v > 0.)
                    .ok_or_else(|| invalid("invalid prediction rotary base"))?
                    as f32;
                if !theta.is_finite() {
                    return Err(invalid("prediction rotary base exceeds f32"));
                }
                Some(PredictionGeometry {
                    layers: schedule(mtp.get("layer_types"), depth, 1)?,
                    rope_theta: theta,
                })
            }
        };
        let vision = if boolean(root, "language_model_only", false)? {
            None
        } else {
            root.get("vision_config")
                .map(|value| {
                    let mut declaration = value.clone();
                    if value
                        .get("deepstack_visual_indexes")
                        .and_then(Value::as_array)
                        .is_some_and(|v| !v.is_empty())
                    {
                        return Err(invalid(
                            "qwen4_exp vision does not declare DeepStack injection",
                        ));
                    }
                    declaration
                        .as_object_mut()
                        .ok_or_else(|| invalid("vision_config must be an object"))?
                        .entry("deepstack_visual_indexes")
                        .or_insert_with(|| serde_json::json!([]));
                    serde_json::from_value::<VisionConfigSource>(declaration)
                        .map_err(|e| invalid(e.to_string()))?
                        .normalize_qwen3_vl()
                        .map_err(|e| invalid(e.to_string()))
                })
                .transpose()?
        };
        if vision
            .as_ref()
            .is_some_and(|vision| vision.out_hidden_size != hidden_size)
        {
            return Err(invalid(
                "vision output width differs from base hidden width",
            ));
        }
        let media = if vision.is_some() {
            let token = |name: &str| {
                unsigned(root, name, None).and_then(|id| {
                    if id < vocabulary as u64 {
                        Ok(id as u32)
                    } else {
                        Err(invalid(format!("{name} exceeds vocabulary")))
                    }
                })
            };
            Some(MediaTokens {
                image: token("image_token_id")?,
                video: token("video_token_id")?,
                start: token("vision_start_token_id")?,
                end: token("vision_end_token_id")?,
            })
        } else {
            None
        };
        Ok(Self {
            vocabulary,
            hidden_size,
            max_positions: integer(text, "max_position_embeddings")?,
            norm_epsilon: epsilon,
            eos,
            tied_embeddings: boolean(
                text,
                "tie_word_embeddings",
                boolean(root, "tie_word_embeddings", false)?,
            )?,
            layers,
            residual,
            attention,
            recurrent,
            experts,
            ngram,
            prediction,
            vision,
            media,
        })
    }
}

#[cfg(test)]
mod tests;

mod gguf;
