//! Architecture-owned checkpoint contracts for Muse Glimmer.

//!
//! Muse Glimmer owns its multimodal SafeTensors catalog, text-only GGUF
//! catalog, sibling vision-projector GGUF catalog, source aliases, geometry,
//! and affine companion constraints here. Generic checkpoint code evaluates
//! those physical declarations without knowing about text or vision models.

use eredu_checkpoint::{StoredDtype, WeightQuantization};

use std::{
    collections::{BTreeSet, HashMap},
    fmt,
};

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::{vision::VisionConfig, DecoderConfig};
use crate::backend::mlx::runtime::checkpoint::{
    store::{SafetensorsWeightStore, WeightStore},
    validation,
};
use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match super::config_from_hf_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(SafetensorsPlanError::Geometry(error)) => return invalid_geometry(error),
        Err(SafetensorsPlanError::Companion { name, detail }) => {
            return CheckpointValidation::Invalid(vec![CheckpointIssue {
                kind: CheckpointIssueKind::CompanionMismatch,
                detail,
                tensor_name: Some(name),
                tensor_type_code: None,
                metadata_key: Some("quantization_config.quant_method".into()),
            }]);
        }
    };
    let mut issues = validation_issues(validation::validate_safetensors_plan(store, &plan));
    append_unexpected_safetensors(store, &plan, "Muse-Glimmer SafeTensors", &mut issues);
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn safetensors_plan(
    args: &DecoderConfig,
) -> Result<SafetensorsCheckpointPlan, SafetensorsPlanError> {
    let hidden = dimension(args.hidden_size, "text hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let intermediate = dimension(args.intermediate_size, "text intermediate size")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        dimension(args.head_dim, "attention head dimension")?,
        "key/value projection width",
    )?;
    let quantization = args.weight_quantization();
    let mut tensors = Vec::new();
    let root = "model.language_model";
    add_safe_matrix(
        &mut tensors,
        &format!("{root}.embed_tokens.weight"),
        vec![vocab, hidden],
        quantization,
    )?;
    tensors.push(safe(format!("{root}.norm.weight"), vec![hidden]));
    add_safe_matrix(
        &mut tensors,
        "lm_head.weight",
        vec![vocab, hidden],
        quantization,
    )?;
    for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
        let block = format!("{root}.layers.{layer}");
        for local in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            tensors.push(safe(format!("{block}.{local}"), vec![hidden]));
        }
        for (local, shape) in [
            ("self_attn.q_proj.weight", vec![query, hidden]),
            ("self_attn.k_proj.weight", vec![key_value, hidden]),
            ("self_attn.v_proj.weight", vec![key_value, hidden]),
            ("self_attn.o_proj.weight", vec![hidden, query]),
            ("self_attn.gate_proj.weight", vec![query, hidden]),
            ("mlp.gate_proj.weight", vec![intermediate, hidden]),
            ("mlp.up_proj.weight", vec![intermediate, hidden]),
            ("mlp.down_proj.weight", vec![hidden, intermediate]),
        ] {
            add_safe_matrix(
                &mut tensors,
                &format!("{block}.{local}"),
                shape,
                quantization,
            )?;
        }
    }

    let vision = args
        .vision_config
        .as_ref()
        .ok_or_else(|| "Muse-Glimmer config has no normalized vision geometry".to_string())?;
    add_safe_vision(&mut tensors, vision, quantization)?;
    let projector = dimension(args.projector_hidden_size, "projector hidden size")?;
    let vision_out = dimension(args.vision_out_hidden_size, "vision output width")?;
    for (name, shape) in [
        (
            "model.vision_adapter.fc1.weight",
            vec![projector, vision_out],
        ),
        (
            "model.vision_adapter.fc2.weight",
            vec![projector, projector],
        ),
        ("model.vision_projection.weight", vec![hidden, projector]),
    ] {
        add_safe_matrix(&mut tensors, name, shape, quantization)?;
    }
    SafetensorsCheckpointPlan::new(
        "Muse-Glimmer SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| SafetensorsPlanError::Geometry(error.to_string()))
}

fn add_safe_vision(
    tensors: &mut Vec<SafetensorsTensorConstraint>,
    vision: &VisionConfig,
    quantization: Option<WeightQuantization>,
) -> Result<(), SafetensorsPlanError> {
    let hidden = dimension(vision.hidden_size, "vision hidden size")?;
    let intermediate = dimension(vision.intermediate_size, "vision intermediate size")?;
    let patch = dimension(vision.patch_size, "vision patch size")?;
    let temporal = dimension(vision.temporal_patch_size, "vision temporal patch size")?;
    let patch_area = checked_mul(patch, patch, "vision patch area")?;
    let patch_input = checked_mul(
        checked_mul(temporal, 3, "vision temporal channel width")?,
        patch_area,
        "vision patch input width",
    )?;
    let positions = checked_mul(
        dimension(vision.pos_height, "vision position height")?,
        dimension(vision.pos_width, "vision position width")?,
        "vision position count",
    )?;
    add_safe_matrix(
        tensors,
        "model.vision_tower.patch_embedder.patch_embedding.weight",
        vec![hidden, patch_input],
        quantization,
    )?;
    add_safe_matrix(
        tensors,
        "model.vision_tower.patch_embedder.position_embedding_table.weight",
        vec![positions, hidden],
        quantization,
    )?;
    for local in [
        "ln_pre.weight",
        "ln_pre.bias",
        "ln_post.weight",
        "ln_post.bias",
    ] {
        tensors.push(safe(format!("model.vision_tower.{local}"), vec![hidden]));
    }
    for layer in 0..vision.layer_count() {
        let block = format!("model.vision_tower.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            tensors.extend([
                safe(format!("{block}.{norm}.weight"), vec![hidden]),
                safe(format!("{block}.{norm}.bias"), vec![hidden]),
            ]);
        }
        for projection in ["q_proj", "k_proj", "v_proj", "proj"] {
            add_safe_matrix(
                tensors,
                &format!("{block}.attn.{projection}.weight"),
                vec![hidden, hidden],
                quantization,
            )?;
            tensors.push(safe(
                format!("{block}.attn.{projection}.bias"),
                vec![hidden],
            ));
        }
        add_safe_matrix(
            tensors,
            &format!("{block}.mlp.fc1.weight"),
            vec![intermediate, hidden],
            quantization,
        )?;
        tensors.push(safe(format!("{block}.mlp.fc1.bias"), vec![intermediate]));
        add_safe_matrix(
            tensors,
            &format!("{block}.mlp.fc2.weight"),
            vec![hidden, intermediate],
            quantization,
        )?;
        tensors.push(safe(format!("{block}.mlp.fc2.bias"), vec![hidden]));
    }
    Ok(())
}

fn add_safe_matrix(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<(), SafetensorsPlanError> {
    output.extend(safe_matrix_constraints(name, shape, quantization)?);
    Ok(())
}

fn safe_matrix_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: Option<WeightQuantization>,
) -> Result<Vec<SafetensorsTensorConstraint>, SafetensorsPlanError> {
    let Some(quantization) = quantization else {
        return Ok(vec![safe(name, shape)]);
    };
    let input = *shape
        .last()
        .ok_or_else(|| format!("quantized Muse-Glimmer matrix {name:?} has scalar shape"))?;
    let bits = dimension(quantization.bits(), "quantization bit width")?;
    let group = dimension(quantization.group_size(), "quantization group size")?;
    let packed_bits = input
        .checked_mul(bits)
        .ok_or_else(|| SafetensorsPlanError::Companion {
            name: name.into(),
            detail: format!("quantized tensor {name:?} packing geometry overflows"),
        })?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(SafetensorsPlanError::Companion {
            name: name.into(),
            detail: format!(
                "quantized tensor {name:?} input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
            ),
        });
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion = shape;
    *companion.last_mut().expect("matrix shape") = input / group;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    let dtype = || {
        StoredDtypeConstraint::OneOf(vec![
            StoredDtype::F16,
            StoredDtype::BF16,
            StoredDtype::F32,
            StoredDtype::U8,
        ])
    };
    let mut constraints = vec![SafetensorsTensorConstraint::required(
        name,
        packed,
        StoredDtypeConstraint::Exact(StoredDtype::U32),
    )
    .with_aliases([format!("{prefix}.inner.weight")])];
    constraints.push(
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion.clone(),
            dtype(),
        )
        .companion(),
    );
    if quantization.has_biases() {
        constraints.push(
            SafetensorsTensorConstraint::required(format!("{prefix}.biases"), companion, dtype())
                .companion(),
        );
    }
    Ok(constraints)
}

fn safe(key: impl Into<String>, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

#[derive(Debug)]
pub(crate) enum SafetensorsPlanError {
    Geometry(String),
    Companion { name: String, detail: String },
}

impl fmt::Display for SafetensorsPlanError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Geometry(detail) | Self::Companion { detail, .. } => formatter.write_str(detail),
        }
    }
}

impl From<String> for SafetensorsPlanError {
    fn from(detail: String) -> Self {
        Self::Geometry(detail)
    }
}

fn append_unexpected_safetensors(
    store: &SafetensorsWeightStore,
    plan: &SafetensorsCheckpointPlan,
    loader: &str,
    issues: &mut Vec<CheckpointIssue>,
) {
    let allowed = plan
        .common_tensors
        .iter()
        .flat_map(|tensor| std::iter::once(&tensor.key).chain(&tensor.aliases))
        .collect::<BTreeSet<_>>();
    for key in store.keys() {
        if !allowed.contains(&key) {
            issues.push(unexpected(&key, loader));
        }
    }
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let translate = |name: &str| super::translate_gguf_weight_name(name, false);
    if let Err(error) = checkpoint.catalog().translated_outputs(translate) {
        return conflicting(error.to_string());
    }
    let args = match super::config_from_gguf_catalog(checkpoint, metadata, "muse-glimmer", false) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validate_strict_gguf(checkpoint, &plan, "Muse-Glimmer GGUF")
}

pub(crate) fn gguf_plan(args: &DecoderConfig) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "text hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let intermediate = dimension(args.intermediate_size, "text intermediate size")?;
    let head = dimension(args.head_dim, "attention head dimension")?;
    let query = checked_mul(
        dimension(args.num_attention_heads, "attention head count")?,
        head,
        "query projection width",
    )?;
    let key_value = checked_mul(
        dimension(args.num_key_value_heads, "key/value head count")?,
        head,
        "key/value projection width",
    )?;
    let mut tensors = vec![
        gguf(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf("output_norm.weight", vec![hidden], TensorOperation::Vector),
        gguf(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
    ];
    for layer in 0..dimension(args.num_hidden_layers, "text layer count")? {
        let block = format!("blk.{layer}");
        for (local, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "post_attention_norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
            (
                "post_ffw_norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            ("attn_q_norm.weight", vec![head], TensorOperation::Vector),
            ("attn_k_norm.weight", vec![head], TensorOperation::Vector),
            (
                "attn_q.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_k.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_v.weight",
                vec![key_value, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_output.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "attn_gate.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_gate.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_up.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "ffn_down.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ] {
            tensors.push(gguf(format!("{block}.{local}"), shape, operation));
        }
    }
    GgufCheckpointPlan::new(
        "Muse-Glimmer GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn validate_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let text = match super::config_from_gguf_catalog(
        model_checkpoint,
        model_metadata,
        "muse-glimmer",
        false,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let vision = match VisionConfig::from_gguf_metadata(metadata, text.hidden_size) {
        Ok(config) => config,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(super::translate_mmproj_weight_name)
    {
        return conflicting(error.to_string());
    }
    let plan = match projector_gguf_plan(&text, &vision) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validate_strict_gguf(checkpoint, &plan, "Muse-Glimmer projector GGUF")
}

pub(crate) fn projector_gguf_plan(
    text: &DecoderConfig,
    vision: &VisionConfig,
) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(vision.hidden_size, "vision hidden size")?;
    let intermediate = dimension(vision.intermediate_size, "vision intermediate size")?;
    let patch = dimension(vision.patch_size, "vision patch size")?;
    let merge = dimension(vision.merge_size, "vision merge size")?;
    let merged = checked_mul(
        hidden,
        checked_mul(merge, merge, "vision merge area")?,
        "merged vision width",
    )?;
    let mut tensors = vec![
        gguf(
            "v.patch_embd.weight",
            vec![hidden, 3, patch, patch],
            TensorOperation::Dense,
        ),
        gguf(
            "v.position_embd.weight",
            vec![1024, hidden],
            TensorOperation::Dense,
        ),
    ];
    for name in [
        "v.pre_ln.weight",
        "v.pre_ln.bias",
        "v.post_ln.weight",
        "v.post_ln.bias",
    ] {
        tensors.push(gguf(name, vec![hidden], TensorOperation::Dense));
    }
    for layer in 0..vision.layer_count() {
        let block = format!("v.blk.{layer}");
        for name in ["ln1.weight", "ln1.bias", "ln2.weight", "ln2.bias"] {
            tensors.push(gguf(
                format!("{block}.{name}"),
                vec![hidden],
                TensorOperation::Dense,
            ));
        }
        for projection in ["attn_q", "attn_k", "attn_v", "attn_out"] {
            tensors.push(gguf(
                format!("{block}.{projection}.weight"),
                vec![hidden, hidden],
                TensorOperation::Matrix,
            ));
            tensors.push(gguf(
                format!("{block}.{projection}.bias"),
                vec![hidden],
                TensorOperation::Dense,
            ));
        }
        tensors.extend([
            gguf(
                format!("{block}.ffn_up.weight"),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.ffn_up.bias"),
                vec![intermediate],
                TensorOperation::Dense,
            ),
            gguf(
                format!("{block}.ffn_down.weight"),
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
            gguf(
                format!("{block}.ffn_down.bias"),
                vec![hidden],
                TensorOperation::Dense,
            ),
        ]);
    }
    tensors.extend([
        gguf("mm.0.weight", vec![4096, merged], TensorOperation::Matrix),
        gguf("mm.1.weight", vec![4096, 4096], TensorOperation::Matrix),
        gguf(
            "mm.2.weight",
            vec![dimension(text.hidden_size, "text hidden size")?, 4096],
            TensorOperation::Matrix,
        ),
    ]);
    GgufCheckpointPlan::new(
        "Muse-Glimmer projector GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

fn validate_strict_gguf(
    checkpoint: &GgufCheckpoint,
    plan: &GgufCheckpointPlan,
    loader: &str,
) -> CheckpointValidation {
    let mut issues = validation_issues(validation::validate_gguf_plan(checkpoint, plan));
    let allowed = plan
        .common_tensors
        .iter()
        .flat_map(|tensor| std::iter::once(&tensor.key).chain(&tensor.aliases))
        .collect::<BTreeSet<_>>();
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected(name, loader));
        }
    }
    CheckpointValidation::from_issues(issues)
}

fn gguf(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Muse-Glimmer {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Muse-Glimmer {name} geometry overflows"))
}

fn validation_issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    }
}

fn conflicting(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::ConflictingLayout,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

fn unexpected(name: &str, loader: &str) -> CheckpointIssue {
    CheckpointIssue {
        kind: CheckpointIssueKind::UnexpectedTensor,
        detail: format!("{loader} catalog contains unexpected tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn invalid_geometry(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args() -> DecoderConfig {
        super::super::config_from_hf_value(&serde_json::json!({
            "architectures": ["MuseGlimmerForConditionalGeneration"],
            "model_type": "muse_glimmer",
            "image_token_id": 30, "video_token_id": 29,
            "out_hidden_size": 16, "projector_hidden_size": 8,
            "vision_config": {
                "model_type": "muse_glimmer_vision", "hidden_act": "gelu",
                "hidden_size": 4, "intermediate_size": 8,
                "num_attention_heads": 1, "num_hidden_layers": 1,
                "patch_size": 2, "patch_temporal": 2, "merge_size": 2,
                "pos_emb_height": 2, "pos_emb_width": 2,
                "max_position_embeddings": 4, "layer_norm_eps": 1e-5,
                "layer_types": ["full_attention"],
                "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"}
            },
            "text_config": {
                "model_type": "muse_glimmer_text", "hidden_size": 24,
                "num_hidden_layers": 1, "intermediate_size": 48,
                "num_attention_heads": 4, "num_key_value_heads": 2,
                "head_dim": 4, "vocab_size": 32, "max_position_embeddings": 128,
                "rms_norm_eps": 1e-5, "post_norm_eps": 1e-8,
                "rope_parameters": {"rope_theta": 500000.0, "rope_type": "default"},
                "layer_types": ["full_attention"], "layer_rope_theta": [0.0],
                "sliding_window": 16, "tie_word_embeddings": false,
                "hidden_activation": "silu", "attention_dropout": 0.0,
                "attention_bias": false, "mlp_bias": false,
                "qk_scale_factor": 3.87, "output_multiplier": 0.19611613513818404,
                "final_logit_softcapping": 20.0
            }
        }))
        .unwrap()
    }

    #[test]
    fn source_plans_keep_hf_vision_and_gguf_qk_norms_distinct() {
        let args = args();
        let safe = safetensors_plan(&args).unwrap();
        let safe_names = safe
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(safe_names.contains("model.vision_tower.layers.0.attn.q_proj.weight"));
        assert!(safe_names.contains("model.language_model.layers.0.self_attn.gate_proj.weight"));
        assert!(!safe_names.contains("model.language_model.layers.0.self_attn.q_norm.weight"));

        let gguf = gguf_plan(&args).unwrap();
        let gguf_names = gguf
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(gguf_names.contains("blk.0.attn_q_norm.weight"));
        assert!(!gguf_names.iter().any(|name| name.starts_with("v.")));

        let projector = projector_gguf_plan(
            &args,
            args.vision_config
                .as_ref()
                .expect("normalized vision config"),
        )
        .unwrap();
        assert!(projector
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "v.blk.0.attn_q.weight"));
        assert!(projector
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "mm.2.weight" && tensor.shape == [24, 4096]));
    }
}
