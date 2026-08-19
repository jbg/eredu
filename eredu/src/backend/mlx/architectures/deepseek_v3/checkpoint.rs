//! Architecture-owned checkpoint contracts for DeepSeek-V3/R1 and DeepSeek2 GGUF.

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, LayerPolicy, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::{
    contract::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation},
    quantization::WeightQuantization,
    schema::{
        AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
        GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan, SafetensorsTensorConstraint,
        StoredDtypeConstraint, TensorOperation,
    },
    store::{SafetensorsWeightStore, StoredDtype},
    validation,
};

#[derive(Clone, Copy)]
enum MatrixFormat {
    Dense,
    Affine(WeightQuantization),
    Fp8,
}

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    allow_packed_experts: bool,
) -> CheckpointValidation {
    let args = match model::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match safetensors_plan(&args, allow_packed_experts) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(
    args: &ModelArgs,
    allow_packed_experts: bool,
) -> Result<SafetensorsCheckpointPlan, String> {
    let affine = args
        .affine_quantization()
        .map_err(|error| error.to_string())?;
    let native_fp8 = args.native_fp8_config().is_some();
    let format_for = |name: &str, operation: TensorOperation| {
        if operation != TensorOperation::Matrix
            || name.ends_with(".mlp.gate.weight")
            || (native_fp8 && (name == "model.embed_tokens.weight" || name == "lm_head.weight"))
        {
            MatrixFormat::Dense
        } else if native_fp8 {
            MatrixFormat::Fp8
        } else if let Some(affine) = affine {
            MatrixFormat::Affine(affine)
        } else {
            MatrixFormat::Dense
        }
    };
    let mut common = Vec::new();
    let mut groups = Vec::new();
    add_root(args, &mut common, &format_for)?;
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        add_layer(
            args,
            layer,
            policy,
            &mut common,
            &mut groups,
            allow_packed_experts,
            &format_for,
        )?;
    }
    for index in 0..count(args.num_nextn_predict_layers, "MTP layer count")? {
        let global = checked_add(
            dimension(args.num_hidden_layers, "layer count")?,
            index,
            "MTP layer index",
        )?;
        add_layer(
            args,
            global,
            LayerPolicy::SparseMoe,
            &mut common,
            &mut groups,
            allow_packed_experts,
            &format_for,
        )?;
        let prefix = format!("model.layers.{global}");
        let hidden = dimension(args.hidden_size, "hidden size")?;
        let vocab = dimension(args.vocab_size, "vocab size")?;
        for (name, shape, operation) in [
            ("enorm.weight", vec![hidden], TensorOperation::Vector),
            ("hnorm.weight", vec![hidden], TensorOperation::Vector),
            (
                "eh_proj.weight",
                vec![hidden, checked_mul(2, hidden, "MTP input width")?],
                TensorOperation::Matrix,
            ),
            (
                "shared_head.norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "shared_head.head.weight",
                vec![vocab, hidden],
                TensorOperation::Matrix,
            ),
        ] {
            add_constraint(
                &mut common,
                &format!("{prefix}.{name}"),
                shape,
                format_for(&format!("{prefix}.{name}"), operation),
            )?;
        }
    }
    SafetensorsCheckpointPlan::new(
        "DeepSeek-V3 SafeTensors",
        common,
        groups,
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn add_root(
    args: &ModelArgs,
    common: &mut Vec<SafetensorsTensorConstraint>,
    format_for: &impl Fn(&str, TensorOperation) -> MatrixFormat,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocab size")?;
    for (name, shape, operation) in [
        (
            "model.embed_tokens.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        ("model.norm.weight", vec![hidden], TensorOperation::Vector),
        (
            "lm_head.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
    ] {
        add_constraint(common, name, shape, format_for(name, operation))?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn add_layer(
    args: &ModelArgs,
    layer: usize,
    policy: LayerPolicy,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    allow_packed_experts: bool,
    format_for: &impl Fn(&str, TensorOperation) -> MatrixFormat,
) -> Result<(), String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let query_head = checked_add(
        dimension(args.qk_nope_head_dim, "query no-PE width")?,
        dimension(args.qk_rope_head_dim, "query RoPE width")?,
        "query head width",
    )?;
    let prefix = format!("model.layers.{layer}");
    let mut entries = vec![
        (
            "input_layernorm.weight".to_string(),
            vec![hidden],
            TensorOperation::Vector,
        ),
        (
            "post_attention_layernorm.weight".to_string(),
            vec![hidden],
            TensorOperation::Vector,
        ),
    ];
    if let Some(rank) = args.q_lora_rank {
        let rank = dimension(rank, "query LoRA rank")?;
        entries.extend([
            (
                "self_attn.q_a_proj.weight".into(),
                vec![rank, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.q_a_layernorm.weight".into(),
                vec![rank],
                TensorOperation::Vector,
            ),
            (
                "self_attn.q_b_proj.weight".into(),
                vec![checked_mul(heads, query_head, "query projection")?, rank],
                TensorOperation::Matrix,
            ),
        ]);
    } else {
        entries.push((
            "self_attn.q_proj.weight".into(),
            vec![checked_mul(heads, query_head, "query projection")?, hidden],
            TensorOperation::Matrix,
        ));
    }
    let kv_rank = dimension(args.kv_lora_rank, "KV LoRA rank")?;
    let rope = dimension(args.qk_rope_head_dim, "query RoPE width")?;
    let nope = dimension(args.qk_nope_head_dim, "query no-PE width")?;
    let value = dimension(args.v_head_dim, "value head width")?;
    entries.extend([
        (
            "self_attn.kv_a_proj_with_mqa.weight".into(),
            vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
            TensorOperation::Matrix,
        ),
        (
            "self_attn.kv_a_layernorm.weight".into(),
            vec![kv_rank],
            TensorOperation::Vector,
        ),
        (
            "self_attn.kv_b_proj.weight".into(),
            vec![
                checked_mul(
                    heads,
                    checked_add(nope, value, "KV-B head width")?,
                    "KV-B width",
                )?,
                kv_rank,
            ],
            TensorOperation::Matrix,
        ),
        (
            "self_attn.o_proj.weight".into(),
            vec![hidden, checked_mul(heads, value, "attention output width")?],
            TensorOperation::Matrix,
        ),
    ]);
    if policy == LayerPolicy::SparseMoe {
        let experts = dimension(args.n_routed_experts, "expert count")?;
        let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
        let shared = checked_mul(
            intermediate,
            dimension(args.n_shared_experts, "shared expert count")?,
            "shared expert width",
        )?;
        entries.extend([
            (
                "mlp.gate.weight".into(),
                vec![experts, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.gate.e_score_correction_bias".into(),
                vec![experts],
                TensorOperation::Vector,
            ),
            (
                "mlp.shared_experts.gate_proj.weight".into(),
                vec![shared, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.shared_experts.up_proj.weight".into(),
                vec![shared, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.shared_experts.down_proj.weight".into(),
                vec![hidden, shared],
                TensorOperation::Matrix,
            ),
        ]);
        let expert_format = format_for(
            &format!("{prefix}.mlp.experts.gate_proj"),
            TensorOperation::Matrix,
        );
        for (projection, split_shape, packed_shape) in [
            (
                "gate_proj",
                vec![intermediate, hidden],
                vec![experts, intermediate, hidden],
            ),
            (
                "up_proj",
                vec![intermediate, hidden],
                vec![experts, intermediate, hidden],
            ),
            (
                "down_proj",
                vec![hidden, intermediate],
                vec![experts, hidden, intermediate],
            ),
        ] {
            groups.push(expert_group(
                &format!("{prefix}.mlp.experts"),
                projection,
                experts,
                split_shape,
                packed_shape,
                allow_packed_experts,
                expert_format,
            )?);
        }
    } else {
        let intermediate = dimension(args.intermediate_size, "intermediate size")?;
        entries.extend([
            (
                "mlp.gate_proj.weight".into(),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.up_proj.weight".into(),
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.down_proj.weight".into(),
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ]);
    }
    for (local, shape, operation) in entries {
        let name = format!("{prefix}.{local}");
        add_constraint(common, &name, shape, format_for(&name, operation))?;
    }
    Ok(())
}

fn expert_group(
    prefix: &str,
    projection: &str,
    experts: usize,
    split_shape: Vec<usize>,
    packed_shape: Vec<usize>,
    allow_packed: bool,
    format: MatrixFormat,
) -> Result<AlternativeLayoutGroup<SafetensorsTensorConstraint>, String> {
    let mut split = Vec::new();
    let mut split_keys = Vec::new();
    for expert in 0..experts {
        let name = format!("{prefix}.{expert}.{projection}.weight");
        split_keys.push(name.clone());
        split.extend(format_constraints(&name, split_shape.clone(), format)?);
    }
    let mut variants = vec![LayoutVariant {
        id: "split".into(),
        tensors: split,
        discriminator_keys: split_keys,
    }];
    if allow_packed {
        let packed = format!("{prefix}.{projection}");
        variants.push(LayoutVariant {
            id: "packed".into(),
            tensors: format_constraints(&packed, packed_shape, format)?,
            discriminator_keys: vec![packed],
        });
    }
    Ok(AlternativeLayoutGroup {
        id: format!("{prefix} {projection} storage"),
        required: true,
        variants,
    })
}

fn add_constraint(
    output: &mut Vec<SafetensorsTensorConstraint>,
    name: &str,
    shape: Vec<usize>,
    format: MatrixFormat,
) -> Result<(), String> {
    output.extend(format_constraints(name, shape, format)?);
    Ok(())
}

fn format_constraints(
    name: &str,
    shape: Vec<usize>,
    format: MatrixFormat,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    match format {
        MatrixFormat::Dense => Ok(vec![SafetensorsTensorConstraint::required(
            name,
            shape,
            StoredDtypeConstraint::Floating,
        )]),
        MatrixFormat::Fp8 => {
            if shape.len() < 2 {
                return Err(format!("FP8 tensor {name:?} must have rank at least two"));
            }
            let mut scale_shape = shape.clone();
            let rank = scale_shape.len();
            scale_shape[rank - 2] = scale_shape[rank - 2].div_ceil(128);
            scale_shape[rank - 1] = scale_shape[rank - 1].div_ceil(128);
            Ok(vec![
                SafetensorsTensorConstraint::required(
                    name,
                    shape,
                    StoredDtypeConstraint::OneOf(vec![StoredDtype::F8E4M3, StoredDtype::U8]),
                ),
                SafetensorsTensorConstraint::required(
                    fp8_scale_name(name),
                    scale_shape,
                    companion_dtype(),
                )
                .companion(),
            ])
        }
        MatrixFormat::Affine(quantization) => affine_constraints(name, shape, quantization),
    }
}

fn affine_constraints(
    name: &str,
    shape: Vec<usize>,
    quantization: WeightQuantization,
) -> Result<Vec<SafetensorsTensorConstraint>, String> {
    let input = *shape
        .last()
        .ok_or_else(|| "empty matrix shape".to_string())?;
    let bits = quantization.bits() as usize;
    let group = quantization.group_size() as usize;
    let packed_bits = checked_mul(input, bits, "affine packing")?;
    if !input.is_multiple_of(group) || !input.is_multiple_of(32) || !packed_bits.is_multiple_of(32)
    {
        return Err(format!(
            "quantized input dimension {input} is incompatible with group size {group} and {bits}-bit packing"
        ));
    }
    let mut packed = shape.clone();
    *packed.last_mut().expect("matrix shape") = packed_bits / 32;
    let mut companion_shape = shape;
    *companion_shape.last_mut().expect("matrix shape") = input / group;
    let prefix = name.strip_suffix(".weight").unwrap_or(name);
    let aliases = name
        .strip_suffix(".weight")
        .map(|prefix| vec![format!("{prefix}.inner.weight")])
        .unwrap_or_default();
    let mut result = vec![
        SafetensorsTensorConstraint::required(
            name,
            packed,
            StoredDtypeConstraint::Exact(StoredDtype::U32),
        )
        .with_aliases(aliases),
        SafetensorsTensorConstraint::required(
            format!("{prefix}.scales"),
            companion_shape.clone(),
            companion_dtype(),
        )
        .companion(),
    ];
    if quantization.has_biases() {
        result.push(
            SafetensorsTensorConstraint::required(
                format!("{prefix}.biases"),
                companion_shape,
                companion_dtype(),
            )
            .companion(),
        );
    }
    Ok(result)
}

fn companion_dtype() -> StoredDtypeConstraint {
    StoredDtypeConstraint::OneOf(vec![
        StoredDtype::F16,
        StoredDtype::BF16,
        StoredDtype::F32,
        StoredDtype::U8,
    ])
}

fn fp8_scale_name(name: &str) -> String {
    name.strip_suffix(".weight").map_or_else(
        || format!("{name}_scale_inv"),
        |prefix| format!("{prefix}.weight_scale_inv"),
    )
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let args = match model::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocab size")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let query_head = checked_add(
        dimension(args.qk_nope_head_dim, "query no-PE width")?,
        dimension(args.qk_rope_head_dim, "query RoPE width")?,
        "query head width",
    )?;
    let mut tensors = vec![
        gguf_tensor(
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        gguf_tensor("output_norm.weight", vec![hidden], TensorOperation::Vector),
        gguf_tensor(
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
    ];
    for (layer, policy) in args.layer_schedule.iter().copied().enumerate() {
        let block = format!("blk.{layer}");
        for (name, shape, operation) in [
            ("attn_norm.weight", vec![hidden], TensorOperation::Vector),
            ("ffn_norm.weight", vec![hidden], TensorOperation::Vector),
        ] {
            tensors.push(gguf_tensor(format!("{block}.{name}"), shape, operation));
        }
        if let Some(rank) = args.q_lora_rank {
            let rank = dimension(rank, "query LoRA rank")?;
            for (name, shape, operation) in [
                (
                    "attn_q_a.weight",
                    vec![rank, hidden],
                    TensorOperation::Matrix,
                ),
                ("attn_q_a_norm.weight", vec![rank], TensorOperation::Vector),
                (
                    "attn_q_b.weight",
                    vec![checked_mul(heads, query_head, "query width")?, rank],
                    TensorOperation::Matrix,
                ),
            ] {
                tensors.push(gguf_tensor(format!("{block}.{name}"), shape, operation));
            }
        } else {
            tensors.push(gguf_tensor(
                format!("{block}.attn_q.weight"),
                vec![checked_mul(heads, query_head, "query width")?, hidden],
                TensorOperation::Matrix,
            ));
        }
        let kv_rank = dimension(args.kv_lora_rank, "KV LoRA rank")?;
        let rope = dimension(args.qk_rope_head_dim, "RoPE width")?;
        let nope = dimension(args.qk_nope_head_dim, "no-PE width")?;
        let value = dimension(args.v_head_dim, "value width")?;
        tensors.extend([
            gguf_tensor(
                format!("{block}.attn_kv_a_mqa.weight"),
                vec![checked_add(kv_rank, rope, "KV-A width")?, hidden],
                TensorOperation::Matrix,
            ),
            gguf_tensor(
                format!("{block}.attn_kv_a_norm.weight"),
                vec![kv_rank],
                TensorOperation::Vector,
            ),
            gguf_tensor(
                format!("{block}.attn_output.weight"),
                vec![hidden, checked_mul(heads, value, "output width")?],
                TensorOperation::Matrix,
            ),
        ]);
        if args.split_kv_b {
            tensors.extend([
                gguf_tensor(
                    format!("{block}.attn_k_b.weight"),
                    vec![heads, kv_rank, nope],
                    TensorOperation::Matrix,
                ),
                gguf_tensor(
                    format!("{block}.attn_v_b.weight"),
                    vec![heads, value, kv_rank],
                    TensorOperation::Matrix,
                ),
            ]);
        } else {
            tensors.push(gguf_tensor(
                format!("{block}.attn_kv_b.weight"),
                vec![
                    checked_mul(
                        heads,
                        checked_add(nope, value, "KV-B head width")?,
                        "KV-B width",
                    )?,
                    kv_rank,
                ],
                TensorOperation::Matrix,
            ));
        }
        if policy == LayerPolicy::SparseMoe {
            let experts = dimension(args.n_routed_experts, "expert count")?;
            let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
            let shared = checked_mul(
                intermediate,
                dimension(args.n_shared_experts, "shared expert count")?,
                "shared expert width",
            )?;
            for (name, shape, operation) in [
                (
                    "ffn_gate_inp.weight",
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                ("exp_probs_b.bias", vec![experts], TensorOperation::Vector),
                (
                    "ffn_gate_shexp.weight",
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_up_shexp.weight",
                    vec![shared, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_down_shexp.weight",
                    vec![hidden, shared],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_gate_exps.weight",
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_up_exps.weight",
                    vec![experts, intermediate, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "ffn_down_exps.weight",
                    vec![experts, hidden, intermediate],
                    TensorOperation::Matrix,
                ),
            ] {
                tensors.push(gguf_tensor(format!("{block}.{name}"), shape, operation));
            }
        } else {
            let intermediate = dimension(args.intermediate_size, "intermediate size")?;
            for (name, shape) in [
                ("ffn_gate.weight", vec![intermediate, hidden]),
                ("ffn_up.weight", vec![intermediate, hidden]),
                ("ffn_down.weight", vec![hidden, intermediate]),
            ] {
                tensors.push(gguf_tensor(
                    format!("{block}.{name}"),
                    shape,
                    TensorOperation::Matrix,
                ));
            }
        }
    }
    let mut policy = CatalogPolicy::strict();
    policy.allowed_prefixes.push("rope_freqs.".into());
    GgufCheckpointPlan::new("DeepSeek2 GGUF", tensors, Vec::new(), policy)
        .map_err(|error| error.to_string())
}

fn gguf_tensor(
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
        .ok_or_else(|| format!("DeepSeek-V3 {name} must be positive, got {value}"))
}

fn count(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .map_err(|_| format!("DeepSeek-V3 {name} must be non-negative, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("DeepSeek-V3 {name} geometry overflows"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("DeepSeek-V3 {name} geometry overflows"))
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
