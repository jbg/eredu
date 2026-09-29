//! Complete SafeTensors physical contracts, separate from executable admission.
use super::SafetensorsTableSourcePlan;
use crate::qwen4_exp::config::{Config, LayerKind};
use eredu_checkpoint::{
    schema::{
        matrix_for_linear_format, AlternativeLayoutGroup, CatalogPolicy, LayoutVariant,
        MatrixScaleNames, SafetensorsCheckpointPlan, SafetensorsTensorConstraint as Constraint,
        StoredDtypeConstraint,
    },
    BlockFp8Format, BlockFp8ScaleEncoding, LinearFormat, StoredDtype,
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

/// Declared physical encoding and exact module exclusions, retained by preparation.
#[derive(Debug, Clone)]
pub struct SafetensorsEncoding {
    block: Option<BlockFp8Format>,
    excluded: BTreeSet<String>,
}

/// Normalizes only the released namespace aliases, never arbitrary suffix matches.
pub fn canonical_name(name: &str) -> String {
    if let Some(relative) = name.strip_prefix("model.language_model.") {
        format!("model.{relative}")
    } else if name.starts_with("layers.")
        || name.starts_with("embed_tokens.")
        || name.starts_with("hyper_connection_mixer.")
    {
        format!("model.{name}")
    } else {
        name.into()
    }
}
pub(in crate::qwen4_exp) fn aliases(name: &str) -> Vec<String> {
    if let Some(relative) = name
        .strip_prefix("model.")
        .filter(|s| !s.starts_with("visual."))
    {
        vec![format!("model.language_model.{relative}"), relative.into()]
    } else {
        Vec::new()
    }
}
impl SafetensorsEncoding {
    /// Parses the published block-FP8 policy without routing table scales through it.
    pub fn from_json(root: &Value) -> Result<Self, String> {
        let Some(q) = root.get("quantization_config").filter(|v| !v.is_null()) else {
            return Ok(Self {
                block: None,
                excluded: BTreeSet::new(),
            });
        };
        if q.get("quant_method").and_then(Value::as_str) != Some("fp8")
            || q.get("activation_scheme").and_then(Value::as_str) != Some("dynamic")
            || q.get("weight_block_size") != Some(&serde_json::json!([128, 128]))
            || q.get("weight_per_tensor").and_then(Value::as_bool) != Some(false)
            || q.get("act_per_tensor").and_then(Value::as_bool) != Some(false)
        {
            return Err("qwen4_exp requires the released dynamic 128x128 E4M3 policy".into());
        }
        let excluded = q
            .get("modules_to_not_convert")
            .and_then(Value::as_array)
            .ok_or("FP8 policy omitted exact module exclusions")?
            .iter()
            .map(|v| {
                let name = v
                    .as_str()
                    .filter(|s| !s.is_empty() && !s.contains('*'))
                    .ok_or("FP8 exclusions must be literal module names")?;
                Ok(canonical_name(name))
            })
            .collect::<Result<BTreeSet<_>, String>>()?;
        if q.get("modules_to_convert")
            != Some(&serde_json::json!(["ple.ple_embedding.ngram_embedding"]))
        {
            return Err("FP8 policy must declare the separate n-gram table conversion".into());
        }
        Ok(Self {
            block: Some(
                BlockFp8Format::new(128, 128, BlockFp8ScaleEncoding::FloatingPoint)
                    .map_err(|e| e.to_string())?,
            ),
            excluded,
        })
    }
    /// Exact parameter format; exclusions apply to a module and its descendants.
    pub fn linear_format(&self, parameter: &str) -> LinearFormat {
        let canonical = canonical_name(parameter);
        let module = canonical.strip_suffix(".weight").unwrap_or(&canonical);
        if self.excluded.iter().any(|name| {
            module == name
                || module
                    .strip_prefix(name)
                    .is_some_and(|suffix| suffix.starts_with('.'))
        }) {
            LinearFormat::Dense
        } else {
            self.block
                .map_or(LinearFormat::Dense, LinearFormat::E4M3BlockFp8)
        }
    }
    /// Table rows use one shared scalar, independent of expert blocks.
    pub fn scalar_fp8_table(&self) -> bool {
        self.block.is_some()
    }
}

fn dense(tensors: &mut Vec<Constraint>, name: String, shape: Vec<usize>) {
    tensors.push(
        Constraint::required(&name, shape, StoredDtypeConstraint::Floating)
            .with_aliases(aliases(&name)),
    );
}
fn matrix(
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    name: String,
    shape: Vec<usize>,
) -> Result<(), String> {
    let scale = format!("{name}_scale_inv");
    tensors.extend(
        matrix_for_linear_format(
            &name,
            aliases(&name),
            shape,
            encoding.linear_format(&name),
            Some(MatrixScaleNames {
                key: scale.clone(),
                aliases: aliases(&scale),
            }),
        )
        .map_err(|e| e.to_string())?,
    );
    Ok(())
}
fn residual(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    root: &str,
    injection: bool,
) -> Result<(), String> {
    let width = config.hidden_size as usize * config.residual.streams as usize;
    dense(tensors, format!("{root}.hc_norm.weight"), vec![width]);
    for (suffix, shape) in [
        (
            "input_mix_weight_down",
            vec![config.residual.rank as usize, width],
        ),
        (
            "input_mix_weight_up",
            vec![width, config.residual.rank as usize],
        ),
    ] {
        matrix(encoding, tensors, format!("{root}.{suffix}.weight"), shape)?;
    }
    if injection {
        matrix(
            encoding,
            tensors,
            format!("{root}.block_inject_weight.weight"),
            vec![config.residual.streams as usize, width],
        )?;
    }
    Ok(())
}
fn block(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    groups: &mut Vec<AlternativeLayoutGroup<Constraint>>,
    root: &str,
    kind: LayerKind,
) -> Result<(), String> {
    let h = config.hidden_size as usize;
    for branch in ["attn_hyper_connection", "mlp_hyper_connection"] {
        residual(config, encoding, tensors, &format!("{root}.{branch}"), true)?;
    }
    match kind {
        LayerKind::Recurrent => {
            let r = &config.recurrent;
            let k = r.key_heads as usize * r.key_dim as usize;
            let v = r.value_heads as usize * r.value_dim as usize;
            let prefix = format!("{root}.linear_attn");
            for (name, shape) in [
                ("in_proj_qkv", vec![2 * k + v, h]),
                ("in_proj_z", vec![v, h]),
                ("in_proj_a", vec![r.value_heads as usize, h]),
                ("in_proj_b", vec![r.value_heads as usize, h]),
                ("out_proj", vec![h, v]),
            ] {
                matrix(encoding, tensors, format!("{prefix}.{name}.weight"), shape)?;
            }
            dense(
                tensors,
                format!("{prefix}.conv1d.weight"),
                vec![2 * k + v, 1, r.kernel as usize],
            );
            for name in ["A_log", "dt_bias"] {
                dense(
                    tensors,
                    format!("{prefix}.{name}"),
                    vec![r.value_heads as usize],
                );
            }
            dense(
                tensors,
                format!("{prefix}.norm.weight"),
                vec![r.value_dim as usize],
            );
        }
        LayerKind::Indexed => {
            let a = &config.attention;
            let q = a.heads as usize * a.head_dim as usize;
            let kv = a.kv_heads as usize * a.head_dim as usize;
            let prefix = format!("{root}.self_attn");
            for (name, width) in [("q_proj", 2 * q), ("k_proj", kv), ("v_proj", kv)] {
                matrix(
                    encoding,
                    tensors,
                    format!("{prefix}.{name}.weight"),
                    vec![width, h],
                )?;
                if a.bias {
                    dense(tensors, format!("{prefix}.{name}.bias"), vec![width]);
                }
            }
            matrix(
                encoding,
                tensors,
                format!("{prefix}.o_proj.weight"),
                vec![h, q],
            )?;
            for name in ["q_norm", "k_norm"] {
                dense(
                    tensors,
                    format!("{prefix}.{name}.weight"),
                    vec![a.head_dim as usize],
                );
            }
            matrix(
                encoding,
                tensors,
                format!("{prefix}.indexer.index_qk_proj.weight"),
                vec![
                    (a.index_heads as usize + a.index_kv_heads as usize)
                        * a.index_head_dim as usize,
                    h,
                ],
            )?;
            for name in ["q_layernorm", "k_layernorm"] {
                dense(
                    tensors,
                    format!("{prefix}.indexer.{name}.weight"),
                    vec![a.index_head_dim as usize],
                );
            }
        }
    }
    let e = &config.experts;
    let count = e.count as usize;
    let inner = e.intermediate as usize;
    let shared = e.shared_intermediate as usize;
    let root = format!("{root}.mlp");
    for (name, shape) in [
        ("gate", vec![count, h]),
        ("shared_expert_gate", vec![1, h]),
        ("shared_expert.gate_proj", vec![shared, h]),
        ("shared_expert.up_proj", vec![shared, h]),
        ("shared_expert.down_proj", vec![h, shared]),
    ] {
        matrix(encoding, tensors, format!("{root}.{name}.weight"), shape)?;
    }
    let mut packed = Vec::new();
    let mut separate = Vec::new();
    matrix(
        encoding,
        &mut packed,
        format!("{root}.experts.gate_up_proj"),
        vec![count, 2 * inner, h],
    )?;
    matrix(
        encoding,
        &mut packed,
        format!("{root}.experts.down_proj"),
        vec![count, h, inner],
    )?;
    for expert in 0..count {
        for (name, shape) in [
            ("gate_proj", vec![inner, h]),
            ("up_proj", vec![inner, h]),
            ("down_proj", vec![h, inner]),
        ] {
            matrix(
                encoding,
                &mut separate,
                format!("{root}.experts.{expert}.{name}.weight"),
                shape,
            )?;
        }
    }
    groups.push(AlternativeLayoutGroup {
        id: format!("{root}.experts storage"),
        required: true,
        variants: vec![
            LayoutVariant {
                id: "packed".into(),
                discriminator_keys: packed.iter().map(|t| t.key.clone()).collect(),
                tensors: packed,
            },
            LayoutVariant {
                id: "separate".into(),
                discriminator_keys: separate.iter().map(|t| t.key.clone()).collect(),
                tensors: separate,
            },
        ],
    });
    Ok(())
}

/// Builds the full target, embedded prediction, vision and exact physical table
/// catalog. Table plans validate headers only; literal hash values are acquired
/// after the complete artifact has passed this contract.
pub fn safetensors_plan(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tables: &BTreeMap<usize, SafetensorsTableSourcePlan>,
) -> Result<SafetensorsCheckpointPlan, String> {
    if tables.keys().copied().collect::<Vec<_>>() != config.ngram.layers {
        return Err("prepared tables do not match injection layers".into());
    }
    safetensors_plan_inner(config, encoding, Some(tables))
}

/// Preliminary configuration contract. Physical table geometry is added only
/// when the actual catalog is admitted; this schema cannot authorize loading.
pub(crate) fn safetensors_config_plan(
    config: &Config,
    encoding: &SafetensorsEncoding,
) -> Result<SafetensorsCheckpointPlan, String> {
    safetensors_plan_inner(config, encoding, None)
}

fn safetensors_plan_inner(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tables: Option<&BTreeMap<usize, SafetensorsTableSourcePlan>>,
) -> Result<SafetensorsCheckpointPlan, String> {
    let h = config.hidden_size as usize;
    let width = h * config.residual.streams as usize;
    let mut tensors = Vec::new();
    let mut groups = Vec::new();
    matrix(
        encoding,
        &mut tensors,
        "model.embed_tokens.weight".into(),
        vec![config.vocabulary as usize, h],
    )?;
    if !config.tied_embeddings {
        matrix(
            encoding,
            &mut tensors,
            "lm_head.weight".into(),
            vec![config.vocabulary as usize, h],
        )?;
    }
    residual(
        config,
        encoding,
        &mut tensors,
        "model.hyper_connection_mixer",
        false,
    )?;
    for (layer, kind) in config.layers.iter().copied().enumerate() {
        let root = format!("model.layers.{layer}");
        block(config, encoding, &mut tensors, &mut groups, &root, kind)?;
        if config.ngram.layers.contains(&layer) {
            let table = tables.and_then(|tables| tables.get(&layer));
            if table.is_some_and(|table| table.scalar_fp8() != encoding.scalar_fp8_table()) {
                return Err("table scalar encoding differs from the checkpoint policy".into());
            }
            for name in ["norm_conv", "norm_key", "norm_query"] {
                dense(
                    &mut tensors,
                    format!("{root}.ple.{name}.weight"),
                    vec![width],
                );
            }
            dense(
                &mut tensors,
                format!("{root}.ple.conv1d.weight"),
                vec![width, 1, config.ngram.kernel as usize],
            );
            for (name, out) in [("key_proj", width), ("value_proj", h)] {
                matrix(
                    encoding,
                    &mut tensors,
                    format!("{root}.ple.{name}.weight"),
                    vec![out, config.ngram.embedding_dim as usize],
                )?;
            }
            let prefix = format!("{root}.ple.ple_embedding");
            for (name, count) in [
                ("layer_multipliers", config.ngram.order as usize),
                (
                    "ngram_heads_offsets",
                    ((config.ngram.order - 1) * config.ngram.heads) as usize,
                ),
                (
                    "ngram_heads_vocab_sizes",
                    ((config.ngram.order - 1) * config.ngram.heads) as usize,
                ),
            ] {
                let name = format!("{prefix}.{name}");
                tensors.push(
                    Constraint::required(
                        &name,
                        vec![count],
                        StoredDtypeConstraint::Exact(StoredDtype::I64),
                    )
                    .with_aliases(aliases(&name)),
                );
            }
            if let Some(table) = table {
                for name in table.shards.iter().chain(table.scale_name.iter()) {
                    let meta = &table.catalog[name];
                    let canonical = canonical_name(name);
                    tensors.push(
                        Constraint::required(
                            &canonical,
                            meta.shape.clone(),
                            StoredDtypeConstraint::Exact(meta.stored_dtype.clone()),
                        )
                        .with_aliases(aliases(&canonical)),
                    );
                }
            }
        }
    }
    prediction_tensors(config, encoding, &mut tensors, &mut groups)?;
    if let Some(vision) = &config.vision {
        let vision = crate::qwen::vision::safetensors_plan(vision, "model.visual")?;
        // Vision is explicitly excluded by the released FP8 conversion policy.
        for tensor in &vision.common_tensors {
            if tensor.shape.len() == 2
                && tensor.key.ends_with(".weight")
                && encoding.linear_format(&tensor.key) != LinearFormat::Dense
            {
                return Err(format!(
                    "visual parameter {} must retain its declared dense encoding",
                    tensor.key
                ));
            }
        }
        tensors.extend(vision.common_tensors);
        groups.extend(vision.layout_groups);
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("qwen4_exp SafeTensors", tensors, groups, policy)
        .map_err(|e| e.to_string())
}

fn prediction_tensors(
    config: &Config,
    encoding: &SafetensorsEncoding,
    tensors: &mut Vec<Constraint>,
    groups: &mut Vec<AlternativeLayoutGroup<Constraint>>,
) -> Result<(), String> {
    let h = config.hidden_size as usize;
    let width = h * config.residual.streams as usize;
    if let Some(prediction) = &config.prediction {
        dense(tensors, "mtp.pre_fc_norm_hidden.weight".into(), vec![width]);
        dense(tensors, "mtp.pre_fc_norm_embedding.weight".into(), vec![h]);
        for name in ["fc_hidden", "fc_embedding"] {
            matrix(encoding, tensors, format!("mtp.{name}.weight"), vec![h, h])?;
        }
        residual(
            config,
            encoding,
            tensors,
            "mtp.hyper_connection_mixer",
            false,
        )?;
        for (layer, kind) in prediction.layers.iter().copied().enumerate() {
            block(
                config,
                encoding,
                tensors,
                groups,
                &format!("mtp.layers.{layer}"),
                kind,
            )?;
        }
    }
    Ok(())
}

/// Exact prediction role, usable for a standalone MTP artifact or a restricted
/// view of a complete official checkpoint. Vocabulary weights belong to the target.
pub fn prediction_safetensors_plan(
    config: &Config,
    encoding: &SafetensorsEncoding,
) -> Result<SafetensorsCheckpointPlan, String> {
    if config
        .prediction
        .as_ref()
        .is_none_or(|p| p.layers.is_empty())
    {
        return Err("prediction source must declare at least one depth".into());
    }
    let mut tensors = Vec::new();
    let mut groups = Vec::new();
    prediction_tensors(config, encoding, &mut tensors, &mut groups)?;
    let mut policy = CatalogPolicy::strict();
    policy.allowed_suffixes.push("rotary_emb.inv_freq".into());
    SafetensorsCheckpointPlan::new("qwen4_exp prediction SafeTensors", tensors, groups, policy)
        .map_err(|e| e.to_string())
}
