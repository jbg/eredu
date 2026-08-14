//! Architecture-owned checkpoint contracts for Qwen3-VL text and vision weights.

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::{model, vision::VisionConfig};
use crate::{
    architectures::qwen::dense::{self, checkpoint as dense_checkpoint},
    runtime::checkpoint::{
        contract::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation},
        schema::{
            CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
            SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint,
            TensorOperation,
        },
        store::{SafetensorsWeightStore, WeightStore},
        validation,
    },
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub(crate) enum GgufVariant {
    Dense,
    Moe,
}

impl GgufVariant {
    const fn metadata_name(self) -> &'static str {
        match self {
            Self::Dense => "qwen3vl",
            Self::Moe => "qwen3vlmoe",
        }
    }

    const fn is_moe(self) -> bool {
        matches!(self, Self::Moe)
    }
}

pub(crate) fn validate_safetensors(
    expected_moe: bool,
    config: &Value,
    store: &SafetensorsWeightStore,
    allow_derived_expert_layouts: bool,
) -> CheckpointValidation {
    let args = match model::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let is_moe = args.text_config.is_moe();
    if is_moe != expected_moe {
        return invalid_geometry(format!(
            "Qwen3-VL dispatch selected {}, but the nested text configuration is {}",
            if expected_moe { "MoE" } else { "dense" },
            if is_moe { "MoE" } else { "dense" }
        ));
    }
    if args.text_config.num_hidden_layers as usize > store.keys().len()
        || args.vision_config.layer_count() > store.keys().len()
    {
        return invalid_geometry(format!(
            "configured Qwen3-VL text/vision depths {}/{} exceed the entire {}-tensor checkpoint catalog",
            args.text_config.num_hidden_layers,
            args.vision_config.layer_count(),
            store.keys().len()
        ));
    }
    let plan = match safetensors_plan(&args, allow_derived_expert_layouts) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(
    args: &model::ModelArgs,
    allow_derived_expert_layouts: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let text = dense_checkpoint::safetensors_plan_with_root(
        &args.text_config,
        "model.language_model",
        allow_derived_expert_layouts,
    )?;
    let mut common = text.common_tensors;
    common.extend(vision_safetensors_constraints(
        &args.vision_config,
        dimension(args.text_config.hidden_size, "text hidden_size")?,
        "model.visual",
    )?);
    SafetensorsCheckpointPlan::new(
        "Qwen3-VL SafeTensors",
        common,
        text.layout_groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn vision_safetensors_constraints(
    config: &VisionConfig,
    text_hidden: usize,
    root: &str,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let hidden = dimension(config.hidden_size, "vision hidden size")?;
    let intermediate = dimension(config.intermediate_size, "vision intermediate size")?;
    let channels = dimension(config.in_channels, "vision channel count")?;
    let temporal = dimension(config.temporal_patch_size, "vision temporal patch size")?;
    let patch = dimension(config.patch_size, "vision patch size")?;
    let merge = dimension(config.spatial_merge_size, "vision merge size")?;
    let merger_hidden = checked_mul(
        hidden,
        checked_mul(merge, merge, "vision merge area")?,
        "vision merger width",
    )?;
    let mut tensors = vec![
        vision_tensor(
            format!("{root}.pos_embed.weight"),
            vec![
                dimension(config.num_position_embeddings, "vision positions")?,
                hidden,
            ],
        ),
        vision_tensor(
            format!("{root}.patch_embed.proj.weight"),
            vec![hidden, channels, temporal, patch, patch],
        ),
        vision_tensor(format!("{root}.patch_embed.proj.bias"), vec![hidden]),
    ];
    for layer in 0..config.layer_count() {
        let prefix = format!("{root}.blocks.{layer}");
        for (name, shape) in [
            ("norm1.weight", vec![hidden]),
            ("norm1.bias", vec![hidden]),
            (
                "attn.qkv.weight",
                vec![checked_mul(3, hidden, "vision QKV width")?, hidden],
            ),
            (
                "attn.qkv.bias",
                vec![checked_mul(3, hidden, "vision QKV bias")?],
            ),
            ("attn.proj.weight", vec![hidden, hidden]),
            ("attn.proj.bias", vec![hidden]),
            ("norm2.weight", vec![hidden]),
            ("norm2.bias", vec![hidden]),
            ("mlp.linear_fc1.weight", vec![intermediate, hidden]),
            ("mlp.linear_fc1.bias", vec![intermediate]),
            ("mlp.linear_fc2.weight", vec![hidden, intermediate]),
            ("mlp.linear_fc2.bias", vec![hidden]),
        ] {
            tensors.push(vision_tensor(format!("{prefix}.{name}"), shape));
        }
    }
    for (name, shape) in [
        ("merger.norm.weight", vec![hidden]),
        ("merger.norm.bias", vec![hidden]),
        (
            "merger.linear_fc1.weight",
            vec![merger_hidden, merger_hidden],
        ),
        ("merger.linear_fc1.bias", vec![merger_hidden]),
        ("merger.linear_fc2.weight", vec![text_hidden, merger_hidden]),
        ("merger.linear_fc2.bias", vec![text_hidden]),
    ] {
        tensors.push(vision_tensor(format!("{root}.{name}"), shape));
    }
    for index in 0..config.deepstack_layer_count() {
        let prefix = format!("{root}.deepstack_merger_list.{index}");
        for (name, shape) in [
            ("norm.weight", vec![merger_hidden]),
            ("norm.bias", vec![merger_hidden]),
            ("linear_fc1.weight", vec![merger_hidden, merger_hidden]),
            ("linear_fc1.bias", vec![merger_hidden]),
            ("linear_fc2.weight", vec![text_hidden, merger_hidden]),
            ("linear_fc2.bias", vec![text_hidden]),
        ] {
            tensors.push(vision_tensor(format!("{prefix}.{name}"), shape));
        }
    }
    Ok(tensors)
}

fn vision_tensor(key: String, shape: Vec<usize>) -> SafetensorsTensorConstraint {
    SafetensorsTensorConstraint::required(key, shape, StoredDtypeConstraint::Floating)
}

pub(crate) fn validate_gguf(
    variant: GgufVariant,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let is_moe = variant.is_moe();
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(|name| dense::translate_gguf_weight_name(name, is_moe))
    {
        return conflict(error.to_string());
    }
    let args = match dense::config_from_gguf_catalog(
        checkpoint,
        metadata,
        variant.metadata_name(),
        is_moe,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = model::validate_qwen3_vl_text_gguf_catalog(&args, metadata) {
        return invalid_geometry(error.to_string());
    }
    let dense_variant = if is_moe {
        dense_checkpoint::GgufVariant::Qwen3Moe
    } else {
        dense_checkpoint::GgufVariant::Qwen3
    };
    let dense_plan = match dense_checkpoint::gguf_plan(&args, dense_variant) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    let plan = match GgufCheckpointPlan::new(
        "Qwen3-VL text GGUF",
        dense_plan.common_tensors,
        dense_plan.layout_groups,
        CatalogPolicy::strict(),
    ) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let mut issues = validation_issues(validation::validate_gguf_plan(checkpoint, &plan));
    if is_moe {
        issues.extend(validation::validate_matching_gguf_encodings(
            checkpoint,
            (0..args.num_hidden_layers as usize).map(|layer| {
                (
                    format!("blk.{layer}.ffn_gate_exps.weight"),
                    format!("blk.{layer}.ffn_up_exps.weight"),
                )
            }),
            "Qwen3-VL-MoE",
        ));
    }
    CheckpointValidation::from_issues(issues)
}

pub(crate) fn validate_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let architecture = match dense::gguf_string(model_metadata, "general.architecture") {
        Ok(architecture) => architecture,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let is_moe = architecture == "qwen3vlmoe";
    if architecture != "qwen3vl" && !is_moe {
        return invalid_geometry(format!(
            "Qwen3-VL projector requires qwen3vl or qwen3vlmoe text, got {architecture:?}"
        ));
    }
    let text_args = match dense::config_from_gguf_catalog(
        model_checkpoint,
        model_metadata,
        &architecture,
        is_moe,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match model::qwen3_vl_args_from_gguf_catalog(
        text_args,
        model_metadata,
        checkpoint,
        metadata,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let deepstack = args.vision_config.deepstack_layers();
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(|name| model::translate_qwen3_vl_mmproj_name(name, &deepstack))
    {
        return conflict(error.to_string());
    }
    let plan = match projector_gguf_plan(&args.vision_config, args.text_config.hidden_size) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn projector_gguf_plan(
    vision: &VisionConfig,
    text_hidden_size: i32,
) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(vision.hidden_size, "vision hidden size")?;
    let intermediate = dimension(vision.intermediate_size, "vision intermediate size")?;
    let text_hidden = dimension(text_hidden_size, "text hidden size")?;
    let patch = dimension(vision.patch_size, "vision patch size")?;
    let merge = dimension(vision.spatial_merge_size, "vision merge size")?;
    let merger_hidden = checked_mul(
        hidden,
        checked_mul(merge, merge, "vision merge area")?,
        "vision merger width",
    )?;
    let mut tensors = vec![
        gguf_tensor(
            "v.position_embd.weight",
            vec![
                dimension(vision.num_position_embeddings, "vision positions")?,
                hidden,
            ],
            TensorOperation::Dense,
        ),
        gguf_tensor(
            "v.patch_embd.weight",
            vec![hidden, 3, patch, patch],
            TensorOperation::Dense,
        ),
        gguf_tensor(
            "v.patch_embd.weight.1",
            vec![hidden, 3, patch, patch],
            TensorOperation::Dense,
        ),
        gguf_tensor("v.patch_embd.bias", vec![hidden], TensorOperation::Dense),
    ];
    for layer in 0..vision.layer_count() {
        let prefix = format!("v.blk.{layer}");
        for (name, shape, operation) in [
            ("ln1.weight", vec![hidden], TensorOperation::Dense),
            ("ln1.bias", vec![hidden], TensorOperation::Dense),
            (
                "attn_qkv.weight",
                vec![checked_mul(3, hidden, "vision QKV width")?, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn_qkv.bias",
                vec![checked_mul(3, hidden, "vision QKV bias")?],
                TensorOperation::Dense,
            ),
            (
                "attn_out.weight",
                vec![hidden, hidden],
                TensorOperation::Matrix,
            ),
            ("attn_out.bias", vec![hidden], TensorOperation::Dense),
            ("ln2.weight", vec![hidden], TensorOperation::Dense),
            ("ln2.bias", vec![hidden], TensorOperation::Dense),
            (
                "ffn_up.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            ("ffn_up.bias", vec![intermediate], TensorOperation::Dense),
            (
                "ffn_down.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
            ("ffn_down.bias", vec![hidden], TensorOperation::Dense),
        ] {
            tensors.push(gguf_tensor(format!("{prefix}.{name}"), shape, operation));
        }
    }
    for (name, shape, operation) in [
        ("v.post_ln.weight", vec![hidden], TensorOperation::Dense),
        ("v.post_ln.bias", vec![hidden], TensorOperation::Dense),
        (
            "mm.0.weight",
            vec![merger_hidden, merger_hidden],
            TensorOperation::Matrix,
        ),
        ("mm.0.bias", vec![merger_hidden], TensorOperation::Dense),
        (
            "mm.2.weight",
            vec![text_hidden, merger_hidden],
            TensorOperation::Matrix,
        ),
        ("mm.2.bias", vec![text_hidden], TensorOperation::Dense),
    ] {
        tensors.push(gguf_tensor(name, shape, operation));
    }
    for layer in vision.deepstack_layers() {
        let prefix = format!("v.deepstack.{layer}");
        for (name, shape, operation) in [
            ("norm.weight", vec![merger_hidden], TensorOperation::Dense),
            ("norm.bias", vec![merger_hidden], TensorOperation::Dense),
            (
                "fc1.weight",
                vec![merger_hidden, merger_hidden],
                TensorOperation::Matrix,
            ),
            ("fc1.bias", vec![merger_hidden], TensorOperation::Dense),
            (
                "fc2.weight",
                vec![text_hidden, merger_hidden],
                TensorOperation::Matrix,
            ),
            ("fc2.bias", vec![text_hidden], TensorOperation::Dense),
        ] {
            tensors.push(gguf_tensor(format!("{prefix}.{name}"), shape, operation));
        }
    }
    GgufCheckpointPlan::new(
        "Qwen3-VL projector GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn gguf_tensor(
    key: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> GgufTensorConstraint {
    GgufTensorConstraint::required(key, shape, GgufTypeConstraint::OperationClass(operation))
}

fn validation_issues(validation: CheckpointValidation) -> Vec<CheckpointIssue> {
    match validation {
        CheckpointValidation::Exact => Vec::new(),
        CheckpointValidation::Invalid(issues) => issues,
        CheckpointValidation::Unverified(issue) => vec![issue],
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("Qwen3-VL {name} must be positive, got {value}"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("Qwen3-VL {name} geometry overflows"))
}

fn conflict(detail: String) -> CheckpointValidation {
    CheckpointValidation::Invalid(vec![CheckpointIssue {
        kind: CheckpointIssueKind::ConflictingLayout,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
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
