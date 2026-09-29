//! Published qwen4exp text metadata, llama.cpp 2145525a4081d66ff1a87cf43ef809f95a85ac0c.
use super::*;
use eredu_gguf::{Checkpoint, MetadataArray, MetadataValue};
use serde_json::{json, Map};
use std::collections::BTreeMap;

struct Fields<'a>(&'a BTreeMap<String, MetadataValue>);
impl Fields<'_> {
    fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.0.get(&format!("qwen4exp.{key}"))
    }
    fn integer(&self, key: &str) -> Result<i32, ConfigError> {
        self.get(key)
            .and_then(MetadataValue::as_i64)
            .and_then(|v| i32::try_from(v).ok())
            .filter(|v| *v > 0)
            .ok_or_else(|| invalid(format!("qwen4exp.{key} must be a positive i32 integer")))
    }
    fn float(&self, key: &str) -> Result<f32, ConfigError> {
        self.get(key)
            .and_then(MetadataValue::as_f32)
            .filter(|v| v.is_finite() && *v > 0.)
            .ok_or_else(|| {
                invalid(format!(
                    "qwen4exp.{key} must be positive finite floating metadata"
                ))
            })
    }
    fn integers(&self, key: &str, count: usize) -> Result<Vec<i64>, ConfigError> {
        self.get(key)
            .and_then(MetadataValue::as_array)
            .filter(|v| v.len() == count)
            .and_then(MetadataArray::to_i64_vec)
            .ok_or_else(|| {
                invalid(format!(
                    "qwen4exp.{key} must contain exactly {count} signed-representable integers"
                ))
            })
    }
}

impl Config {
    /// Normalizes the published GGUF text contract using only retained headers.
    /// No sidecar, payload access, tokenizer reconstruction or native context is needed.
    /// The pinned exporter omits prediction weights and uses a separate vision
    /// projector: neither is fabricated here. `eos[0]` is the literal PLE reset;
    /// tokenizer termination metadata remains owned by the shared text/facade policy.
    pub fn from_gguf(checkpoint: &Checkpoint) -> Result<Self, ConfigError> {
        let f = Fields(checkpoint.metadata());
        if f.0
            .get("general.architecture")
            .and_then(MetadataValue::as_str)
            != Some("qwen4exp")
        {
            return Err(invalid("expected GGUF architecture qwen4exp"));
        }
        if f.get("nextn_predict_layers")
            .is_some_and(|v| v.as_i64() != Some(0))
        {
            return Err(invalid("the pinned qwen4exp exporter omits MTP; prediction weights require a matching separate source"));
        }
        if f.get("rope.scaling.type")
            .is_some_and(|v| v.as_str() != Some("none"))
        {
            return Err(invalid(
                "GGUF rotary scaling beyond the released unscaled contract is not implemented",
            ));
        }
        if f.get("attention.causal")
            .is_some_and(|v| v.as_bool() != Some(true))
        {
            return Err(invalid("released qwen4exp attention is causal"));
        }
        if f.get("expert_gating_func")
            .is_some_and(|v| v.as_i64() != Some(1))
        {
            return Err(invalid("released qwen4exp routing requires softmax gating"));
        }
        let count = f.integer("block_count")? as usize;
        // Check before creating schedules, even for malicious tiny headers.
        if count > 4096 {
            return Err(invalid(
                "GGUF layer schedule exceeds the 4096-entry metadata bound",
            ));
        }
        let recurrent = match f.get("attention.recurrent_layers") {
            Some(MetadataValue::Array(MetadataArray::Bool(values))) if values.len() == count => {
                values.clone()
            }
            Some(_) => {
                return Err(invalid(
                    "recurrent_layers must be a boolean array covering every layer",
                ))
            }
            None => {
                let interval = if f.get("full_attention_interval").is_some() {
                    f.integer("full_attention_interval")?
                } else {
                    4
                } as usize;
                (0..count).map(|i| (i + 1) % interval != 0).collect()
            }
        };
        let ratios = f.integers("attention.compress_ratios", count)?;
        let mut ratio = None;
        for (&is_recurrent, &value) in recurrent.iter().zip(&ratios) {
            if is_recurrent {
                if value != 0 {
                    return Err(invalid("recurrent layers must have zero QSA compression"));
                }
            } else {
                let value = i32::try_from(value)
                    .ok()
                    .filter(|v| *v > 0)
                    .ok_or_else(|| invalid("indexed layers require positive QSA compression"))?;
                if ratio
                    .replace(value)
                    .is_some_and(|previous| previous != value)
                {
                    return Err(invalid("layer-varying QSA compression is not implemented"));
                }
            }
        }
        let ratio =
            ratio.ok_or_else(|| invalid("released GGUF schedule requires indexed attention"))?;
        let hidden = f.integer("embedding_length")?;
        let shape = |name: &str| -> Result<Vec<u64>, ConfigError> {
            checkpoint
                .tensors()
                .find(|t| t.descriptor().name == name)
                .map(|t| t.descriptor().row_major_shape())
                .ok_or_else(|| invalid(format!("missing GGUF tensor {name}")))
        };
        let embedding = shape("token_embd.weight")?;
        if embedding.len() != 2 || embedding[1] != hidden as u64 {
            return Err(invalid("token embedding shape differs from hidden width"));
        }
        let vocabulary = i32::try_from(embedding[0])
            .ok()
            .filter(|v| *v > 0)
            .ok_or_else(|| invalid("invalid token vocabulary extent"))?;
        if f.get("vocab_size").is_some() && f.integer("vocab_size")? != vocabulary {
            return Err(invalid("vocab_size differs from token embedding rows"));
        }
        if let Some(tokens) = f.0.get("tokenizer.ggml.tokens") {
            if tokens
                .as_strings()
                .is_none_or(|v| v.len() != vocabulary as usize)
            {
                return Err(invalid("tokenizer token count differs from embedding rows"));
            }
        }
        let output = checkpoint
            .tensors()
            .any(|t| t.descriptor().name == "output.weight");
        if output && shape("output.weight")? != embedding {
            return Err(invalid(
                "output shape differs from token embedding geometry",
            ));
        }
        let head_dim = f.integer("attention.key_length")?;
        if f.get("attention.value_length").is_some()
            && f.integer("attention.value_length")? != head_dim
        {
            return Err(invalid(
                "different attention key/value dimensions are not implemented",
            ));
        }
        let rotary = f.integer("rope.dimension_count")?;
        let sections = f.integers("rope.dimension_sections", 4)?;
        if sections[3] != 0 {
            return Err(invalid(
                "released MRoPE requires three sections and a zero fourth entry",
            ));
        }
        let value_heads = f.integer("ssm.time_step_rank")?;
        let inner = f.integer("ssm.inner_size")?;
        if inner % value_heads != 0 {
            return Err(invalid(
                "recurrent inner size must divide into complete value heads",
            ));
        }
        let table = shape("per_layer_token_embd.weight")?;
        if table.len() != 2 || table[0] == 0 || table[1] == 0 {
            return Err(invalid("invalid GGUF n-gram table shape"));
        }
        let order = f.integer("ple.ngram_size")?;
        let heads = f.integer("ple.heads_per_ngram")?;
        let hash_heads = order
            .checked_sub(1)
            .and_then(|v| v.checked_mul(heads))
            .filter(|v| *v > 0 && *v <= 4096)
            .ok_or_else(|| invalid("invalid bounded n-gram heads"))?;
        let ple_width = i32::try_from(table[1])
            .ok()
            .and_then(|v| v.checked_mul(hash_heads))
            .ok_or_else(|| invalid("n-gram embedding width exceeds i32"))?;
        // The published container has one set of PLE constants, hence one owner.
        let layers = f.integers("ple.layers", 1)?;
        let layer = layers[0];
        if layer < 0 || layer >= count as i64 {
            return Err(invalid("PLE layer is outside the decoder"));
        }
        let reset = f
            .get("ple.eos_token_id")
            .and_then(MetadataValue::as_i64)
            .filter(|v| *v >= 0 && *v < i64::from(vocabulary))
            .ok_or_else(|| invalid("invalid literal PLE reset token"))?;
        let mut text = Map::new();
        for (name, key) in [
            ("num_attention_heads", "attention.head_count"),
            ("num_key_value_heads", "attention.head_count_kv"),
            ("hc_count", "hyper_connection.count"),
            ("hc_lowrank", "hyper_connection.low_rank"),
            ("max_position_embeddings", "context_length"),
            ("indexer_n_heads", "attention.indexer.head_count"),
            ("indexer_head_dim", "attention.indexer.key_length"),
            ("indexer_budget", "attention.indexer.top_k"),
            ("linear_num_key_heads", "ssm.group_count"),
            ("linear_key_head_dim", "ssm.state_size"),
            ("linear_conv_kernel_dim", "ssm.conv_kernel"),
            ("num_experts", "expert_count"),
            ("num_experts_per_tok", "expert_used_count"),
            ("moe_intermediate_size", "expert_feed_forward_length"),
            (
                "shared_expert_intermediate_size",
                "expert_shared_feed_forward_length",
            ),
            ("ple_conv_kernel_size", "ple.conv_kernel"),
        ] {
            text.insert(name.into(), json!(f.integer(key)?));
        }
        for (name, value) in
            [
                ("model_type", json!("qwen4_exp_text")),
                ("hidden_act", json!("silu")),
                // These equations are fixed by qwen4exp's published runner, not optional
                // SafeTensors generation defaults (which would choose SiLU gating).
                ("output_gate_type", json!("sigmoid")),
                ("indexer_kv_heads", json!(1)),
                ("norm_topk_prob", json!(true)),
                ("hidden_size", json!(hidden)),
                ("num_hidden_layers", json!(count)),
                ("vocab_size", json!(vocabulary)),
                ("eos_token_id", json!(reset)),
                ("tie_word_embeddings", json!(!output)),
                ("head_dim", json!(head_dim)),
                ("indexer_compress_ratio", json!(ratio)),
                ("linear_num_value_heads", json!(value_heads)),
                ("linear_value_head_dim", json!(inner / value_heads)),
                (
                    "rms_norm_eps",
                    json!(f.float("attention.layer_norm_rms_epsilon")?),
                ),
                ("ngram_size", json!(order)),
                ("heads_per_ngram", json!(heads)),
                ("ple_embed_dim", json!(ple_width)),
                ("ple_layer_ids", json!([layer + 1])),
                (
                    "layer_types",
                    json!(recurrent
                        .iter()
                        .map(|r| if *r {
                            "linear_attention"
                        } else {
                            "indexed_attention"
                        })
                        .collect::<Vec<_>>()),
                ),
                (
                    "attention_bias",
                    json!(checkpoint.tensors().any(|t| {
                        recurrent.iter().enumerate().any(|(layer, &is_recurrent)| {
                            !is_recurrent
                                && ["attn_q.bias", "attn_k.bias", "attn_v.bias"].iter().any(
                                    |suffix| t.descriptor().name == format!("blk.{layer}.{suffix}"),
                                )
                        })
                    })),
                ),
                (
                    "rope_parameters",
                    json!({"rope_type": "default", "rope_theta": f.float("rope.freq_base")?,
                "partial_rotary_factor": f64::from(rotary) / f64::from(head_dim),
                "mrope_section": &sections[..3], "mrope_interleaved": true}),
                ),
            ]
        {
            text.insert(name.into(), value);
        }
        let config = Self::from_document(
            &Value::Object(text),
            Some(NGramSourceLayout::Gguf { rows: table[0] }),
        )?;
        crate::qwen4_exp::checkpoint::gguf::controls(checkpoint, &config, layer as usize)
            .map_err(|e| invalid(e.to_string()))?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests;
