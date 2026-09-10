use super::{model_args_from_config_value, ConfigError, ModelArgs};
use eredu_gguf::MetadataValue;
use std::collections::HashMap;

/// Resolves the publisher's `k2-horizon` metadata and optional tensor policies.
pub fn model_args_from_gguf_catalog(
    catalog: &impl crate::GgufTensorCatalog,
    metadata: &HashMap<String, MetadataValue>,
) -> Result<ModelArgs, ConfigError> {
    let invalid = |message: String| ConfigError::Invalid(message);
    if metadata
        .get("general.architecture")
        .and_then(MetadataValue::as_str)
        != Some("k2-horizon")
    {
        return Err(invalid("expected GGUF architecture k2-horizon".into()));
    }
    let key = |suffix: &str| format!("k2-horizon.{suffix}");
    let integer = |suffix: &str, default: Option<i64>| -> Result<i64, ConfigError> {
        match metadata.get(&key(suffix)) {
            Some(v) => v
                .as_i64()
                .ok_or_else(|| invalid(format!("GGUF {suffix} must be an integer"))),
            None => default.ok_or_else(|| invalid(format!("GGUF is missing {suffix}"))),
        }
    };
    let number = |suffix: &str, default: Option<f64>| -> Result<f64, ConfigError> {
        match metadata.get(&key(suffix)) {
            Some(v) => v
                .as_f32()
                .map(f64::from)
                .filter(|n| n.is_finite())
                .ok_or_else(|| invalid(format!("GGUF {suffix} must be finite"))),
            None => default.ok_or_else(|| invalid(format!("GGUF is missing {suffix}"))),
        }
    };
    let boolean = |suffix: &str, default: bool| -> Result<bool, ConfigError> {
        match metadata.get(&key(suffix)) {
            Some(MetadataValue::Bool(v)) => Ok(*v),
            None => Ok(default),
            _ => Err(invalid(format!("GGUF {suffix} must be boolean"))),
        }
    };
    let layers = integer("block_count", None)?;
    let leading = integer("leading_dense_block_count", Some(0))?;
    if !(0..=layers).contains(&leading) || layers <= 0 || layers > 1_000_000 {
        return Err(invalid("invalid GGUF dense prefix or block count".into()));
    }
    let vocabulary = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(MetadataValue::as_strings)
        .map(|v| v.len() as i64);
    let head = integer("attention.key_length", None)?;
    if integer("attention.value_length", Some(head))? != head {
        return Err(invalid("K2 key and value head widths must match".into()));
    }
    let has = |suffix: &str| catalog.any(|name| name.starts_with("blk.") && name.ends_with(suffix));
    let kind = metadata
        .get(&key("rope.scaling.type"))
        .and_then(MetadataValue::as_str)
        .unwrap_or("default");
    let mut rope = serde_json::json!({"rope_type":kind});
    if kind == "yarn" {
        rope["factor"] = number("rope.scaling.factor", None)?.into();
        rope["original_max_position_embeddings"] =
            integer("rope.scaling.original_context_length", None)?.into();
        rope["beta_fast"] = number("rope.scaling.yarn_beta_fast", Some(32.0))?.into();
        rope["beta_slow"] = number("rope.scaling.yarn_beta_slow", Some(1.0))?.into();
        if metadata.contains_key(&key("rope.scaling.yarn_attn_factor")) {
            rope["attention_factor"] = number("rope.scaling.yarn_attn_factor", None)?.into();
        }
    } else if kind == "linear" {
        rope["factor"] = number("rope.scaling.factor", None)?.into();
    }
    let scoring = match integer("expert_gating_func", Some(2))? {
        1 => "softmax",
        2 => "sigmoid",
        n => return Err(invalid(format!("unknown expert_gating_func {n}"))),
    };
    let value = serde_json::json!({
        "model_type":"k2_horizon", "hidden_size":integer("embedding_length",None)?,
        "vocab_size":integer("vocab_size",vocabulary)?, "num_hidden_layers":layers,
        "intermediate_size":integer("feed_forward_length",None)?,
        "num_attention_heads":integer("attention.head_count",None)?,
        "num_key_value_heads":integer("attention.head_count_kv",None)?,
        "head_dim":head, "rope_head_dim":integer("rope.dimension_count",Some(head))?,
        "max_position_embeddings":integer("context_length",None)?,
        "rms_norm_eps":number("attention.layer_norm_rms_epsilon",None)?,
        "layernorm_num_groups":integer("attention.group_norm_groups",Some(1))?,
        "rope_theta":number("rope.freq_base",Some(10_000.0))?, "rope_parameters":rope,
        "attention_bias":has(".attn_q.bias"), "query_key_norm":has(".attn_q_norm.weight"),
        "attention_gate_func":has(".attn_gate.weight").then_some("softplus"),
        "tie_word_embeddings":!catalog.contains("output.weight"),
        "num_experts":integer("expert_count",Some(0))?, "num_experts_per_tok":integer("expert_used_count",Some(0))?,
        "num_shared_experts":integer("expert_shared_count",Some(0))?,
        "moe_intermediate_size":integer("expert_feed_forward_length",Some(0))?,
        "mova_num_experts":integer("attention.value_expert_count",Some(0))?,
        "mova_num_experts_per_tok":integer("attention.value_expert_used_count",Some(0))?,
        "moe_gate_bias":has(".exp_probs_b.bias") || has(".attn_v_gate.bias"),
        "norm_topk_prob":boolean("expert_weights_norm",false)?, "router_score_func":scoring,
        "router_scaling_factor":number("expert_weights_scale",Some(1.0))?,
        "decoder_sparse_step":integer("moe_every_n_layers",Some(1))?,
        "mlp_only_layers":(0..leading).collect::<Vec<_>>(),
    });
    let args = model_args_from_config_value(&value)?;
    if let Some(shared) = metadata.get(&key("expert_shared_feed_forward_length")) {
        if shared.as_i64()
            != Some(i64::from(args.moe_intermediate_size) * i64::from(args.num_shared_experts))
        {
            return Err(invalid("inconsistent shared expert width".into()));
        }
    }
    Ok(args)
}
