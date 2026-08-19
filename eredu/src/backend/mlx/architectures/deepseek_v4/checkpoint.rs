//! Architecture-owned checkpoint contracts for the DeepSeek-V4 family.

use eredu_checkpoint::StoredDtype;

use std::collections::HashMap;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::model::{self, ModelArgs};
use crate::backend::mlx::runtime::checkpoint::store::SafetensorsWeightStore;
use eredu_checkpoint::schema::{
    CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint, GgufTypeConstraint,
    SafetensorsCheckpointPlan, SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
};
use eredu_checkpoint::validation;
use eredu_checkpoint::validation::{CheckpointIssue, CheckpointIssueKind, CheckpointValidation};

#[derive(Debug, Clone, Eq, PartialEq)]
struct TensorSpec {
    safetensors_name: String,
    gguf_name: String,
    shape: Vec<usize>,
    operation: TensorOperation,
}

pub(crate) fn validate_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> CheckpointValidation {
    let args = match ModelArgs::from_value(config.clone()) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let plan = match safetensors_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_safetensors_plan(store, &plan)
}

pub(crate) fn safetensors_plan(args: &ModelArgs) -> Result<SafetensorsCheckpointPlan, String> {
    let mut tensors = common_specs(args)?
        .into_iter()
        .map(|spec| {
            let dtype = if spec.operation == TensorOperation::I32 {
                StoredDtypeConstraint::Exact(StoredDtype::I32)
            } else {
                StoredDtypeConstraint::Floating
            };
            SafetensorsTensorConstraint::required(spec.safetensors_name, spec.shape, dtype)
        })
        .collect::<Vec<_>>();
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
    let experts = dimension(args.n_routed_experts, "expert count")?;
    for layer in 0..dimension(args.num_hidden_layers, "layer count")? {
        for expert in 0..experts {
            for (projection, shape) in [
                ("w1", vec![intermediate, hidden]),
                ("w2", vec![hidden, intermediate]),
                ("w3", vec![intermediate, hidden]),
            ] {
                tensors.push(SafetensorsTensorConstraint::required(
                    format!("layers.{layer}.ffn.experts.{expert}.{projection}.weight"),
                    shape,
                    StoredDtypeConstraint::Floating,
                ));
            }
        }
    }
    SafetensorsCheckpointPlan::new(
        "DeepSeek-V4 SafeTensors",
        tensors,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .map_err(|error| error.to_string())
}

pub(crate) fn validate_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> CheckpointValidation {
    let args = match model::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_nextn_predict_layers != 0 {
        return invalid_geometry(
            "canonical DeepSeek-V4 MTP GGUF files are companion artifacts and cannot be loaded as a base model".into(),
        );
    }
    let plan = match gguf_plan(&args) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error),
    };
    validation::validate_gguf_plan(checkpoint, &plan)
}

pub(crate) fn gguf_plan(args: &ModelArgs) -> Result<GgufCheckpointPlan, String> {
    let mut tensors = common_specs(args)?
        .into_iter()
        .map(|spec| {
            GgufTensorConstraint::required(
                spec.gguf_name,
                spec.shape,
                GgufTypeConstraint::OperationClass(spec.operation),
            )
        })
        .collect::<Vec<_>>();
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let intermediate = dimension(args.moe_intermediate_size, "MoE intermediate size")?;
    let experts = dimension(args.n_routed_experts, "expert count")?;
    for layer in 0..dimension(args.num_hidden_layers, "layer count")? {
        let root = format!("blk.{layer}");
        for (name, shape) in [
            ("ffn_gate_exps.weight", vec![experts, intermediate, hidden]),
            ("ffn_down_exps.weight", vec![experts, hidden, intermediate]),
            ("ffn_up_exps.weight", vec![experts, intermediate, hidden]),
        ] {
            tensors.push(GgufTensorConstraint::required(
                format!("{root}.{name}"),
                shape,
                GgufTypeConstraint::OperationClass(TensorOperation::MxFp4Matrix),
            ));
        }
    }
    GgufCheckpointPlan::new(
        "DeepSeek-V4 GGUF",
        tensors,
        Vec::new(),
        CatalogPolicy::strict(),
    )
    .map_err(|error| error.to_string())
}

fn common_specs(args: &ModelArgs) -> Result<Vec<TensorSpec>, String> {
    let hidden = dimension(args.hidden_size, "hidden size")?;
    let vocab = dimension(args.vocab_size, "vocabulary size")?;
    let hc_mult = dimension(args.hc_mult, "hyper-connection multiplier")?;
    let hc_hidden = checked_mul(hc_mult, hidden, "hyper-connection width")?;
    let mix = checked_mul(
        checked_add(2, hc_mult, "hyper-connection mix input")?,
        hc_mult,
        "hyper-connection mix width",
    )?;
    let mut tensors = vec![
        spec(
            "embed.weight",
            "token_embd.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        spec(
            "norm.weight",
            "output_norm.weight",
            vec![hidden],
            TensorOperation::Vector,
        ),
        spec(
            "head.weight",
            "output.weight",
            vec![vocab, hidden],
            TensorOperation::Matrix,
        ),
        spec(
            "hc_head_fn",
            "output_hc_fn.weight",
            vec![hc_mult, hc_hidden],
            TensorOperation::Matrix,
        ),
        spec(
            "hc_head_base",
            "output_hc_base.weight",
            vec![hc_mult],
            TensorOperation::Vector,
        ),
        spec(
            "hc_head_scale",
            "output_hc_scale.weight",
            vec![1],
            TensorOperation::Vector,
        ),
    ];
    let layers = dimension(args.num_hidden_layers, "layer count")?;
    let heads = dimension(args.num_attention_heads, "attention head count")?;
    let head_dim = dimension(args.head_dim, "attention head width")?;
    let q_rank = dimension(args.q_lora_rank, "query LoRA rank")?;
    let o_rank = dimension(args.o_lora_rank, "output LoRA rank")?;
    let o_groups = dimension(args.o_groups, "output group count")?;
    let experts = dimension(args.n_routed_experts, "expert count")?;
    let routed = dimension(args.num_experts_per_tok, "routed experts per token")?;
    let shared = checked_mul(
        dimension(args.moe_intermediate_size, "MoE intermediate size")?,
        dimension(args.n_shared_experts, "shared expert count")?,
        "shared expert width",
    )?;
    let index_heads = dimension(args.index_n_heads, "index head count")?;
    let index_head_dim = dimension(args.index_head_dim, "index head width")?;
    let hash_layers = count(args.num_hash_layers, "hash layer count")?;
    for layer in 0..layers {
        let root = format!("layers.{layer}");
        let gguf = format!("blk.{layer}");
        let ratio = *args
            .compress_ratios
            .get(layer)
            .ok_or_else(|| format!("DeepSeek-V4 layer {layer} has no compression ratio"))?;
        let query_width = checked_mul(heads, head_dim, "query projection width")?;
        let output_a_rows = checked_mul(o_groups, o_rank, "output-A rows")?;
        let output_a_columns = query_width
            .checked_div(o_groups)
            .ok_or_else(|| "DeepSeek-V4 output group count is zero".to_string())?;
        for (safe, physical, shape, operation) in [
            (
                "attn.wq_a.weight",
                "attn_q_a.weight",
                vec![q_rank, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn.q_norm.weight",
                "attn_q_a_norm.weight",
                vec![q_rank],
                TensorOperation::Vector,
            ),
            (
                "attn.wq_b.weight",
                "attn_q_b.weight",
                vec![query_width, q_rank],
                TensorOperation::Matrix,
            ),
            (
                "attn.wkv.weight",
                "attn_kv.weight",
                vec![head_dim, hidden],
                TensorOperation::Matrix,
            ),
            (
                "attn.kv_norm.weight",
                "attn_kv_a_norm.weight",
                vec![head_dim],
                TensorOperation::Vector,
            ),
            (
                "attn.wo_a.weight",
                "attn_output_a.weight",
                vec![output_a_rows, output_a_columns],
                TensorOperation::Matrix,
            ),
            (
                "attn.wo_b.weight",
                "attn_output_b.weight",
                vec![hidden, output_a_rows],
                TensorOperation::Matrix,
            ),
            (
                "attn.attn_sink",
                "attn_sinks.weight",
                vec![heads],
                TensorOperation::Vector,
            ),
            (
                "attn_norm.weight",
                "attn_norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "ffn_norm.weight",
                "ffn_norm.weight",
                vec![hidden],
                TensorOperation::Vector,
            ),
            (
                "hc_attn_fn",
                "hc_attn_fn.weight",
                vec![mix, hc_hidden],
                TensorOperation::Matrix,
            ),
            (
                "hc_attn_base",
                "hc_attn_base.weight",
                vec![mix],
                TensorOperation::Vector,
            ),
            (
                "hc_attn_scale",
                "hc_attn_scale.weight",
                vec![3],
                TensorOperation::Vector,
            ),
            (
                "hc_ffn_fn",
                "hc_ffn_fn.weight",
                vec![mix, hc_hidden],
                TensorOperation::Matrix,
            ),
            (
                "hc_ffn_base",
                "hc_ffn_base.weight",
                vec![mix],
                TensorOperation::Vector,
            ),
            (
                "hc_ffn_scale",
                "hc_ffn_scale.weight",
                vec![3],
                TensorOperation::Vector,
            ),
            (
                "ffn.gate.weight",
                "ffn_gate_inp.weight",
                vec![experts, hidden],
                TensorOperation::Matrix,
            ),
        ] {
            tensors.push(spec(
                format!("{root}.{safe}"),
                format!("{gguf}.{physical}"),
                shape,
                operation,
            ));
        }
        if layer < hash_layers {
            tensors.push(spec(
                format!("{root}.ffn.gate.tid2eid"),
                format!("{gguf}.ffn_gate_tid2eid.weight"),
                vec![vocab, routed],
                TensorOperation::I32,
            ));
        } else {
            tensors.push(spec(
                format!("{root}.ffn.gate.bias"),
                format!("{gguf}.exp_probs_b.bias"),
                vec![experts],
                TensorOperation::Vector,
            ));
        }
        for (safe, physical, shape) in [
            (
                "ffn.shared_experts.w1.weight",
                "ffn_gate_shexp.weight",
                vec![shared, hidden],
            ),
            (
                "ffn.shared_experts.w2.weight",
                "ffn_down_shexp.weight",
                vec![hidden, shared],
            ),
            (
                "ffn.shared_experts.w3.weight",
                "ffn_up_shexp.weight",
                vec![shared, hidden],
            ),
        ] {
            tensors.push(spec(
                format!("{root}.{safe}"),
                format!("{gguf}.{physical}"),
                shape,
                TensorOperation::Matrix,
            ));
        }
        if ratio != 0 {
            let output = checked_mul(
                head_dim,
                if ratio == 4 { 2 } else { 1 },
                "compressor output width",
            )?;
            let ratio = usize::try_from(ratio).map_err(|_| {
                format!("DeepSeek-V4 compression ratio must be non-negative, got {ratio}")
            })?;
            for (safe, physical, shape, operation) in [
                (
                    "attn.compressor.wkv.weight",
                    "attn_compressor_kv.weight",
                    vec![output, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.compressor.wgate.weight",
                    "attn_compressor_gate.weight",
                    vec![output, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.compressor.ape",
                    "attn_compressor_ape.weight",
                    vec![ratio, output],
                    TensorOperation::Dense,
                ),
                (
                    "attn.compressor.norm.weight",
                    "attn_compressor_norm.weight",
                    vec![head_dim],
                    TensorOperation::Vector,
                ),
            ] {
                tensors.push(spec(
                    format!("{root}.{safe}"),
                    format!("{gguf}.{physical}"),
                    shape,
                    operation,
                ));
            }
        }
        if ratio == 4 {
            let index_output = checked_mul(2, index_head_dim, "index compressor output width")?;
            for (safe, physical, shape, operation) in [
                (
                    "attn.indexer.wq_b.weight",
                    "indexer.attn_q_b.weight",
                    vec![
                        checked_mul(index_heads, index_head_dim, "index query width")?,
                        q_rank,
                    ],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.indexer.weights_proj.weight",
                    "indexer.proj.weight",
                    vec![index_heads, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.indexer.compressor.wkv.weight",
                    "indexer_compressor_kv.weight",
                    vec![index_output, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.indexer.compressor.wgate.weight",
                    "indexer_compressor_gate.weight",
                    vec![index_output, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "attn.indexer.compressor.ape",
                    "indexer_compressor_ape.weight",
                    vec![4, index_output],
                    TensorOperation::Dense,
                ),
                (
                    "attn.indexer.compressor.norm.weight",
                    "indexer_compressor_norm.weight",
                    vec![index_head_dim],
                    TensorOperation::Vector,
                ),
            ] {
                tensors.push(spec(
                    format!("{root}.{safe}"),
                    format!("{gguf}.{physical}"),
                    shape,
                    operation,
                ));
            }
        }
    }
    Ok(tensors)
}

fn spec(
    safetensors_name: impl Into<String>,
    gguf_name: impl Into<String>,
    shape: Vec<usize>,
    operation: TensorOperation,
) -> TensorSpec {
    TensorSpec {
        safetensors_name: safetensors_name.into(),
        gguf_name: gguf_name.into(),
        shape,
        operation,
    }
}

fn dimension(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| format!("DeepSeek-V4 {name} must be positive, got {value}"))
}

fn count(value: i32, name: &str) -> Result<usize, String> {
    usize::try_from(value)
        .map_err(|_| format!("DeepSeek-V4 {name} must be non-negative, got {value}"))
}

fn checked_add(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_add(right)
        .ok_or_else(|| format!("DeepSeek-V4 {name} geometry overflows"))
}

fn checked_mul(left: usize, right: usize, name: &str) -> Result<usize, String> {
    left.checked_mul(right)
        .ok_or_else(|| format!("DeepSeek-V4 {name} geometry overflows"))
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
