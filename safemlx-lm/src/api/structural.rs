//! Pure checkpoint-structure plans shared by inspection and high-level loading.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue, GgufType};
use serde_json::Value;

use super::{GgufArchitecture, ModelKind, ModelLoadOptions};
use crate::{
    architectures::{
        deepseek_v3::model as deepseek_v3,
        deepseek_v4::model as deepseek_v4,
        gemma4::model as gemma4,
        gpt_oss::model as gpt_oss,
        inkling::model as inkling,
        kimi_linear::model as kimi_linear,
        lfm2::model as lfm2,
        llama::model as llama,
        moshi::personaplex,
        muse_glimmer,
        nemotron_h::model as nemotron_h,
        qwen::{
            dense as dense_qwen,
            hybrid::{checkpoint as qwen_hybrid_checkpoint, qwen3_5 as qwen35},
            vl::model as qwen3_vl,
        },
    },
    error::Error,
    runtime::{
        attention::AttentionPolicy,
        checkpoint::{
            schema::{
                gguf_encoding_supported, AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan,
                GgufTensorConstraint, GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
                SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
            },
            store::{SafetensorsWeightStore, StoredDtype, WeightStore},
            validation as checkpoint_validation,
        },
    },
};

pub(crate) use crate::runtime::checkpoint::contract::{
    CheckpointIssue as StructuralIssue, CheckpointIssueKind as StructuralIssueKind,
    CheckpointValidation as StructuralValidation,
};

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
#[allow(dead_code)] // Reserved for fail-closed structural policies.
pub(crate) enum StructuralValidationPolicy {
    Exact,
    Unverified,
}

/// Exhaustive policy table for high-level SafeTensors loader families.
pub(crate) const fn safetensors_policy(kind: ModelKind) -> StructuralValidationPolicy {
    match kind {
        ModelKind::DeepSeekV3
        | ModelKind::DeepSeekV4
        | ModelKind::Gemma4
        | ModelKind::GptOss
        | ModelKind::Inkling
        | ModelKind::KimiLinear
        | ModelKind::Lfm2
        | ModelKind::Llama
        | ModelKind::MuseGlimmer
        | ModelKind::NemotronH
        | ModelKind::PersonaPlex
        | ModelKind::Qwen2
        | ModelKind::Qwen3
        | ModelKind::Qwen3Next
        | ModelKind::Qwen3Vl
        | ModelKind::Qwen3VlMoe
        | ModelKind::Qwen35 => StructuralValidationPolicy::Exact,
    }
}

/// Exhaustive policy table for concrete GGUF loader architectures.
pub(crate) const fn gguf_policy(architecture: GgufArchitecture) -> StructuralValidationPolicy {
    match architecture {
        GgufArchitecture::Llama
        | GgufArchitecture::Mistral
        | GgufArchitecture::MuseGlimmer
        | GgufArchitecture::DeepSeek2
        | GgufArchitecture::DeepSeek4
        | GgufArchitecture::Lfm2
        | GgufArchitecture::Lfm2Moe
        | GgufArchitecture::GptOss
        | GgufArchitecture::Gemma4
        | GgufArchitecture::Inkling
        | GgufArchitecture::Qwen2
        | GgufArchitecture::Qwen3
        | GgufArchitecture::Qwen3Moe
        | GgufArchitecture::NemotronH
        | GgufArchitecture::NemotronHMoe
        | GgufArchitecture::Qwen35
        | GgufArchitecture::Qwen35Moe
        | GgufArchitecture::Qwen3Next
        | GgufArchitecture::Qwen3Vl
        | GgufArchitecture::Qwen3VlMoe
        | GgufArchitecture::KimiLinear => StructuralValidationPolicy::Exact,
    }
}

pub(crate) fn validate_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match safetensors_policy(kind) {
        StructuralValidationPolicy::Exact => match kind {
            ModelKind::DeepSeekV3 => validate_deepseek_v3_safetensors(config, store, options),
            ModelKind::DeepSeekV4 => validate_deepseek_v4_safetensors(config, store),
            ModelKind::Gemma4 => validate_gemma4_safetensors(config, store, options),
            ModelKind::GptOss => validate_gpt_oss_safetensors(config, store),
            ModelKind::Inkling => validate_inkling_safetensors(config, store),
            ModelKind::KimiLinear => validate_kimi_linear_safetensors(config, store),
            ModelKind::Lfm2 => validate_lfm2_safetensors(config, store, options),
            ModelKind::Llama => validate_llama_safetensors(config, store),
            ModelKind::MuseGlimmer => validate_muse_glimmer_safetensors(config, store),
            ModelKind::NemotronH => validate_nemotron_h_safetensors(config, store),
            ModelKind::PersonaPlex => validate_personaplex_safetensors(config, store),
            ModelKind::Qwen2 => validate_dense_qwen_safetensors(config, store),
            ModelKind::Qwen3 => validate_dense_qwen_safetensors(config, store),
            ModelKind::Qwen3Next => validate_qwen3_next_safetensors(config, store, options),
            ModelKind::Qwen3Vl | ModelKind::Qwen3VlMoe => {
                validate_qwen3_vl_safetensors(kind, config, store, options)
            }
            ModelKind::Qwen35 => validate_qwen35_safetensors(config, store, options),
        },
        StructuralValidationPolicy::Unverified => unverified(kind.model_type_name()),
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

pub(crate) fn validate_safetensors_load_path(
    kind: ModelKind,
    model_dir: &Path,
    options: ModelLoadOptions,
) -> Result<(), Error> {
    let config: Value = serde_json::from_slice(&std::fs::read(model_dir.join("config.json"))?)?;
    let store =
        SafetensorsWeightStore::open(model_dir).map_err(|error| Error::Other(Box::new(error)))?;
    validate_safetensors(kind, &config, &store, options).into_loader_result()
}

pub(crate) fn validate_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let validation = match gguf_policy(architecture) {
        StructuralValidationPolicy::Exact => match architecture {
            GgufArchitecture::DeepSeek2 => validate_deepseek2_gguf(checkpoint, metadata),
            GgufArchitecture::DeepSeek4 => validate_deepseek4_gguf(checkpoint, metadata),
            GgufArchitecture::GptOss => validate_gpt_oss_gguf(checkpoint, metadata),
            GgufArchitecture::Gemma4 => validate_gemma4_gguf(checkpoint, metadata, options),
            GgufArchitecture::Inkling => validate_inkling_gguf(checkpoint, metadata, options),
            GgufArchitecture::Lfm2 | GgufArchitecture::Lfm2Moe => {
                validate_lfm2_gguf(architecture, checkpoint, metadata)
            }
            GgufArchitecture::Llama | GgufArchitecture::Mistral => {
                validate_llama_gguf(checkpoint, metadata)
            }
            GgufArchitecture::MuseGlimmer => validate_muse_glimmer_gguf(checkpoint, metadata),
            GgufArchitecture::NemotronH | GgufArchitecture::NemotronHMoe => {
                validate_nemotron_h_gguf(architecture, checkpoint, metadata, options)
            }
            GgufArchitecture::Qwen2 | GgufArchitecture::Qwen3 | GgufArchitecture::Qwen3Moe => {
                validate_dense_qwen_gguf(architecture, checkpoint, metadata)
            }
            architecture @ (GgufArchitecture::Qwen3Vl | GgufArchitecture::Qwen3VlMoe) => {
                validate_qwen3_vl_gguf(architecture, checkpoint, metadata, options)
            }
            GgufArchitecture::KimiLinear => validate_kimi_linear_gguf(checkpoint, metadata),
            GgufArchitecture::Qwen35
            | GgufArchitecture::Qwen35Moe
            | GgufArchitecture::Qwen3Next => {
                validate_qwen35_gguf(architecture, checkpoint, metadata, options)
            }
        },
        StructuralValidationPolicy::Unverified => unverified(architecture.metadata_name()),
    };
    validation.with_strict_catalog(options.weight_residency.strict_loading())
}

fn unverified(architecture: &str) -> StructuralValidation {
    StructuralValidation::Unverified(StructuralIssue {
        kind: StructuralIssueKind::ValidationUnavailable,
        detail: format!(
            "exact header-only structural validation is not yet implemented for {architecture}"
        ),
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    })
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum SafetensorsMatrixFormat {
    Dense,
    Affine(crate::runtime::checkpoint::quantization::WeightQuantization),
    Fp8Block128,
}

#[derive(Debug, Clone)]
struct ExpectedTensor {
    safetensors_name: String,
    gguf_name: String,
    safetensors_shape: Vec<usize>,
    gguf_shape: Vec<usize>,
    operation: TensorOperation,
}

fn llama_expected(args: &llama::ModelArgs) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let intermediate = args.intermediate_size as usize;
    let query = (args.num_attention_heads * args.head_dim) as usize;
    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.input_layernorm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_attention_layernorm.weight"),
                format!("{gguf}.ffn_norm.weight"),
                hidden,
            ),
            expected(
                format!("{model}.self_attn.q_proj.weight"),
                format!("{gguf}.attn_q.weight"),
                [query, hidden],
            ),
            expected(
                format!("{model}.self_attn.k_proj.weight"),
                format!("{gguf}.attn_k.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.v_proj.weight"),
                format!("{gguf}.attn_v.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.o_proj.weight"),
                format!("{gguf}.attn_output.weight"),
                [hidden, query],
            ),
            expected(
                format!("{model}.mlp.gate_proj.weight"),
                format!("{gguf}.ffn_gate.weight"),
                [intermediate, hidden],
            ),
            expected(
                format!("{model}.mlp.up_proj.weight"),
                format!("{gguf}.ffn_up.weight"),
                [intermediate, hidden],
            ),
            expected(
                format!("{model}.mlp.down_proj.weight"),
                format!("{gguf}.ffn_down.weight"),
                [hidden, intermediate],
            ),
        ]);
        if args.attention_bias {
            tensors.extend([
                expected_vector(
                    format!("{model}.self_attn.q_proj.bias"),
                    format!("{gguf}.attn_q.bias"),
                    query,
                ),
                expected_vector(
                    format!("{model}.self_attn.k_proj.bias"),
                    format!("{gguf}.attn_k.bias"),
                    key_value,
                ),
                expected_vector(
                    format!("{model}.self_attn.v_proj.bias"),
                    format!("{gguf}.attn_v.bias"),
                    key_value,
                ),
                expected_vector(
                    format!("{model}.self_attn.o_proj.bias"),
                    format!("{gguf}.attn_output.bias"),
                    hidden,
                ),
            ]);
        }
        if args.mlp_bias {
            tensors.extend([
                expected_vector(
                    format!("{model}.mlp.gate_proj.bias"),
                    format!("{gguf}.ffn_gate.bias"),
                    intermediate,
                ),
                expected_vector(
                    format!("{model}.mlp.up_proj.bias"),
                    format!("{gguf}.ffn_up.bias"),
                    intermediate,
                ),
                expected_vector(
                    format!("{model}.mlp.down_proj.bias"),
                    format!("{gguf}.ffn_down.bias"),
                    hidden,
                ),
            ]);
        }
    }
    tensors
}

fn dense_qwen_expected(args: &dense_qwen::DecoderConfig) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let query = (args.num_attention_heads * args.head_dim) as usize;
    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
    let head = args.head_dim as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.input_layernorm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_attention_layernorm.weight"),
                format!("{gguf}.ffn_norm.weight"),
                hidden,
            ),
            expected(
                format!("{model}.self_attn.q_proj.weight"),
                format!("{gguf}.attn_q.weight"),
                [query, hidden],
            ),
            expected(
                format!("{model}.self_attn.k_proj.weight"),
                format!("{gguf}.attn_k.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.v_proj.weight"),
                format!("{gguf}.attn_v.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.o_proj.weight"),
                format!("{gguf}.attn_output.weight"),
                [hidden, query],
            ),
        ]);
        if args.qk_norm() {
            tensors.extend([
                expected_vector(
                    format!("{model}.self_attn.q_norm.weight"),
                    format!("{gguf}.attn_q_norm.weight"),
                    head,
                ),
                expected_vector(
                    format!("{model}.self_attn.k_norm.weight"),
                    format!("{gguf}.attn_k_norm.weight"),
                    head,
                ),
            ]);
        }
        if args.qkv_bias() {
            tensors.extend([
                expected_vector(
                    format!("{model}.self_attn.q_proj.bias"),
                    format!("{gguf}.attn_q.bias"),
                    query,
                ),
                expected_vector(
                    format!("{model}.self_attn.k_proj.bias"),
                    format!("{gguf}.attn_k.bias"),
                    key_value,
                ),
                expected_vector(
                    format!("{model}.self_attn.v_proj.bias"),
                    format!("{gguf}.attn_v.bias"),
                    key_value,
                ),
            ]);
        }
        if args.is_moe() {
            let experts = args.num_experts as usize;
            let intermediate = args.moe_intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{model}.mlp.gate.weight"),
                    format!("{gguf}.ffn_gate_inp.weight"),
                    [experts, hidden],
                ),
                expected_rank3(
                    format!("{model}.mlp.experts.gate_proj"),
                    format!("{gguf}.ffn_gate_exps.weight"),
                    [experts, intermediate, hidden],
                ),
                expected_rank3(
                    format!("{model}.mlp.experts.up_proj"),
                    format!("{gguf}.ffn_up_exps.weight"),
                    [experts, intermediate, hidden],
                ),
                expected_rank3(
                    format!("{model}.mlp.experts.down_proj"),
                    format!("{gguf}.ffn_down_exps.weight"),
                    [experts, hidden, intermediate],
                ),
            ]);
        } else {
            let intermediate = args.intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{model}.mlp.gate_proj.weight"),
                    format!("{gguf}.ffn_gate.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{model}.mlp.up_proj.weight"),
                    format!("{gguf}.ffn_up.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{model}.mlp.down_proj.weight"),
                    format!("{gguf}.ffn_down.weight"),
                    [hidden, intermediate],
                ),
            ]);
        }
    }
    tensors
}

fn muse_glimmer_expected(config: &Value) -> Result<Vec<ExpectedTensor>, Error> {
    let args = muse_glimmer::config_from_hf_value(config)?;
    let vision = config.get("vision_config").ok_or_else(|| {
        Error::UnsupportedArchitecture("Muse-Glimmer config is missing vision_config".into())
    })?;
    let vision_i32 = |field: &str| -> Result<usize, Error> {
        vision
            .get(field)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                Error::UnsupportedArchitecture(format!(
                    "Muse-Glimmer vision_config requires positive integer {field}"
                ))
            })
    };
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let query = (args.num_attention_heads * args.head_dim) as usize;
    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
    let intermediate = args.intermediate_size as usize;
    let root = "model.language_model";
    let mut tensors = vec![
        expected(
            format!("{root}.embed_tokens.weight"),
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector(format!("{root}.norm.weight"), "output_norm.weight", hidden),
        expected("lm_head.weight", "output.weight", [vocab, hidden]),
    ];
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("{root}.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.input_layernorm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_attention_layernorm.weight"),
                format!("{gguf}.post_attention_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.pre_feedforward_layernorm.weight"),
                format!("{gguf}.ffn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_feedforward_layernorm.weight"),
                format!("{gguf}.post_ffw_norm.weight"),
                hidden,
            ),
            expected(
                format!("{model}.self_attn.q_proj.weight"),
                format!("{gguf}.attn_q.weight"),
                [query, hidden],
            ),
            expected(
                format!("{model}.self_attn.k_proj.weight"),
                format!("{gguf}.attn_k.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.v_proj.weight"),
                format!("{gguf}.attn_v.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.o_proj.weight"),
                format!("{gguf}.attn_output.weight"),
                [hidden, query],
            ),
            expected(
                format!("{model}.self_attn.gate_proj.weight"),
                format!("{gguf}.attn_gate.weight"),
                [query, hidden],
            ),
            expected(
                format!("{model}.mlp.gate_proj.weight"),
                format!("{gguf}.ffn_gate.weight"),
                [intermediate, hidden],
            ),
            expected(
                format!("{model}.mlp.up_proj.weight"),
                format!("{gguf}.ffn_up.weight"),
                [intermediate, hidden],
            ),
            expected(
                format!("{model}.mlp.down_proj.weight"),
                format!("{gguf}.ffn_down.weight"),
                [hidden, intermediate],
            ),
        ]);
    }

    let vision_hidden = vision_i32("hidden_size")?;
    let vision_intermediate = vision_i32("intermediate_size")?;
    let vision_layers = vision_i32("num_hidden_layers")?;
    let patch = vision_i32("patch_size")?;
    let temporal = vision_i32("patch_temporal")?;
    let pos_height = vision_i32("pos_emb_height")?;
    let pos_width = vision_i32("pos_emb_width")?;
    tensors.extend([
        expected(
            "model.vision_tower.patch_embedder.patch_embedding.weight",
            "",
            [vision_hidden, temporal * 3 * patch * patch],
        ),
        expected(
            "model.vision_tower.patch_embedder.position_embedding_table.weight",
            "",
            [pos_height * pos_width, vision_hidden],
        ),
        expected_vector("model.vision_tower.ln_pre.weight", "", vision_hidden),
        expected_vector("model.vision_tower.ln_pre.bias", "", vision_hidden),
        expected_vector("model.vision_tower.ln_post.weight", "", vision_hidden),
        expected_vector("model.vision_tower.ln_post.bias", "", vision_hidden),
    ]);
    for layer in 0..vision_layers {
        let model = format!("model.vision_tower.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            tensors.push(expected_vector(
                format!("{model}.{norm}.weight"),
                "",
                vision_hidden,
            ));
            tensors.push(expected_vector(
                format!("{model}.{norm}.bias"),
                "",
                vision_hidden,
            ));
        }
        for projection in ["q_proj", "k_proj", "v_proj", "proj"] {
            tensors.push(expected(
                format!("{model}.attn.{projection}.weight"),
                "",
                [vision_hidden, vision_hidden],
            ));
            tensors.push(expected_vector(
                format!("{model}.attn.{projection}.bias"),
                "",
                vision_hidden,
            ));
        }
        tensors.push(expected(
            format!("{model}.mlp.fc1.weight"),
            "",
            [vision_intermediate, vision_hidden],
        ));
        tensors.push(expected_vector(
            format!("{model}.mlp.fc1.bias"),
            "",
            vision_intermediate,
        ));
        tensors.push(expected(
            format!("{model}.mlp.fc2.weight"),
            "",
            [vision_hidden, vision_intermediate],
        ));
        tensors.push(expected_vector(
            format!("{model}.mlp.fc2.bias"),
            "",
            vision_hidden,
        ));
    }
    let out_hidden = config
        .get("out_hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(6144) as usize;
    let projector = config
        .get("projector_hidden_size")
        .and_then(Value::as_u64)
        .unwrap_or(4096) as usize;
    tensors.extend([
        expected(
            "model.vision_adapter.fc1.weight",
            "",
            [projector, out_hidden],
        ),
        expected(
            "model.vision_adapter.fc2.weight",
            "",
            [projector, projector],
        ),
        expected("model.vision_projection.weight", "", [hidden, projector]),
    ]);
    Ok(tensors)
}

fn deepseek_v3_expected(args: &deepseek_v3::ModelArgs) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let heads = args.num_attention_heads as usize;
    let query_head = (args.qk_nope_head_dim + args.qk_rope_head_dim) as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
        expected("lm_head.weight", "output.weight", [vocab, hidden]),
    ];
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let prefix = format!("model.layers.{layer}");
        tensors.extend([
            expected_vector(format!("{prefix}.input_layernorm.weight"), "", hidden),
            expected_vector(
                format!("{prefix}.post_attention_layernorm.weight"),
                "",
                hidden,
            ),
        ]);
        if let Some(rank) = args.q_lora_rank {
            tensors.extend([
                expected(
                    format!("{prefix}.self_attn.q_a_proj.weight"),
                    "",
                    [rank as usize, hidden],
                ),
                expected_vector(
                    format!("{prefix}.self_attn.q_a_layernorm.weight"),
                    "",
                    rank as usize,
                ),
                expected(
                    format!("{prefix}.self_attn.q_b_proj.weight"),
                    "",
                    [heads * query_head, rank as usize],
                ),
            ]);
        } else {
            tensors.push(expected(
                format!("{prefix}.self_attn.q_proj.weight"),
                "",
                [heads * query_head, hidden],
            ));
        }
        tensors.extend([
            expected(
                format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                "",
                [(args.kv_lora_rank + args.qk_rope_head_dim) as usize, hidden],
            ),
            expected_vector(
                format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                "",
                args.kv_lora_rank as usize,
            ),
            expected(
                format!("{prefix}.self_attn.kv_b_proj.weight"),
                "",
                [
                    heads * (args.qk_nope_head_dim + args.v_head_dim) as usize,
                    args.kv_lora_rank as usize,
                ],
            ),
            expected(
                format!("{prefix}.self_attn.o_proj.weight"),
                "",
                [hidden, heads * args.v_head_dim as usize],
            ),
        ]);
        if *policy == deepseek_v3::LayerPolicy::SparseMoe {
            let experts = args.n_routed_experts as usize;
            let shared = (args.moe_intermediate_size * args.n_shared_experts) as usize;
            tensors.extend([
                expected(format!("{prefix}.mlp.gate.weight"), "", [experts, hidden]),
                expected_vector(
                    format!("{prefix}.mlp.gate.e_score_correction_bias"),
                    "",
                    experts,
                ),
                expected(
                    format!("{prefix}.mlp.shared_experts.gate_proj.weight"),
                    "",
                    [shared, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.shared_experts.up_proj.weight"),
                    "",
                    [shared, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.shared_experts.down_proj.weight"),
                    "",
                    [hidden, shared],
                ),
            ]);
        } else {
            let intermediate = args.intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{prefix}.mlp.gate_proj.weight"),
                    "",
                    [intermediate, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.up_proj.weight"),
                    "",
                    [intermediate, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.down_proj.weight"),
                    "",
                    [hidden, intermediate],
                ),
            ]);
        }
    }
    for tensor in &mut tensors {
        tensor.gguf_name = deepseek_v3_gguf_name(&tensor.safetensors_name);
    }
    tensors
}

fn deepseek_v3_gguf_name(name: &str) -> String {
    for (runtime, gguf) in [
        ("model.embed_tokens.weight", "token_embd.weight"),
        ("model.norm.weight", "output_norm.weight"),
        ("lm_head.weight", "output.weight"),
    ] {
        if name == runtime {
            return gguf.into();
        }
    }
    let rest = name
        .strip_prefix("model.layers.")
        .expect("DeepSeek expected tensor belongs to a decoder layer");
    let (layer, parameter) = rest
        .split_once('.')
        .expect("DeepSeek expected tensor has a layer-local name");
    let gguf = match parameter {
        "input_layernorm.weight" => "attn_norm.weight",
        "post_attention_layernorm.weight" => "ffn_norm.weight",
        "self_attn.q_proj.weight" => "attn_q.weight",
        "self_attn.q_a_proj.weight" => "attn_q_a.weight",
        "self_attn.q_a_layernorm.weight" => "attn_q_a_norm.weight",
        "self_attn.q_b_proj.weight" => "attn_q_b.weight",
        "self_attn.kv_a_proj_with_mqa.weight" => "attn_kv_a_mqa.weight",
        "self_attn.kv_a_layernorm.weight" => "attn_kv_a_norm.weight",
        "self_attn.kv_b_proj.weight" => "attn_kv_b.weight",
        "self_attn.o_proj.weight" => "attn_output.weight",
        "mlp.gate.weight" => "ffn_gate_inp.weight",
        "mlp.gate.e_score_correction_bias" => "exp_probs_b.bias",
        "mlp.shared_experts.gate_proj.weight" => "ffn_gate_shexp.weight",
        "mlp.shared_experts.up_proj.weight" => "ffn_up_shexp.weight",
        "mlp.shared_experts.down_proj.weight" => "ffn_down_shexp.weight",
        "mlp.gate_proj.weight" => "ffn_gate.weight",
        "mlp.up_proj.weight" => "ffn_up.weight",
        "mlp.down_proj.weight" => "ffn_down.weight",
        _ => panic!("unmapped DeepSeek expected tensor {name}"),
    };
    format!("blk.{layer}.{gguf}")
}

fn gpt_oss_common_expected(args: &gpt_oss::ModelArgs) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let query = (args.num_attention_heads * args.head_dim) as usize;
    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
    let experts = args.num_local_experts as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
        expected("lm_head.weight", "output.weight", [vocab, hidden]),
    ];
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.input_layernorm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_attention_layernorm.weight"),
                format!("{gguf}.attn_post_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.self_attn.sinks"),
                format!("{gguf}.attn_sinks.weight"),
                args.num_attention_heads as usize,
            ),
            expected(
                format!("{model}.self_attn.q_proj.weight"),
                format!("{gguf}.attn_q.weight"),
                [query, hidden],
            ),
            expected_vector(
                format!("{model}.self_attn.q_proj.bias"),
                format!("{gguf}.attn_q.bias"),
                query,
            ),
            expected(
                format!("{model}.self_attn.k_proj.weight"),
                format!("{gguf}.attn_k.weight"),
                [key_value, hidden],
            ),
            expected_vector(
                format!("{model}.self_attn.k_proj.bias"),
                format!("{gguf}.attn_k.bias"),
                key_value,
            ),
            expected(
                format!("{model}.self_attn.v_proj.weight"),
                format!("{gguf}.attn_v.weight"),
                [key_value, hidden],
            ),
            expected_vector(
                format!("{model}.self_attn.v_proj.bias"),
                format!("{gguf}.attn_v.bias"),
                key_value,
            ),
            expected(
                format!("{model}.self_attn.o_proj.weight"),
                format!("{gguf}.attn_output.weight"),
                [hidden, query],
            ),
            expected_vector(
                format!("{model}.self_attn.o_proj.bias"),
                format!("{gguf}.attn_output.bias"),
                hidden,
            ),
            expected_dense_with_gguf_shape(
                format!("{model}.mlp.router.weight"),
                format!("{gguf}.ffn_gate_inp.weight"),
                vec![experts, hidden],
                vec![experts, hidden],
            ),
            expected_vector(
                format!("{model}.mlp.router.bias"),
                format!("{gguf}.ffn_gate_inp.bias"),
                experts,
            ),
        ]);
    }
    tensors
}

fn gpt_oss_gguf_expected(args: &gpt_oss::ModelArgs) -> Vec<ExpectedTensor> {
    let mut tensors = gpt_oss_common_expected(args);
    for tensor in &mut tensors {
        if tensor.gguf_name.ends_with(".ffn_gate_inp.weight") {
            tensor.operation = TensorOperation::Matrix;
        }
    }
    let hidden = args.hidden_size as usize;
    let intermediate = args.intermediate_size as usize;
    let experts = args.num_local_experts as usize;
    for layer in 0..args.num_hidden_layers as usize {
        let gguf = format!("blk.{layer}");
        for projection in ["gate", "up"] {
            tensors.push(expected_mxfp4_rank3(
                format!("{gguf}.ffn_{projection}_exps.weight"),
                [experts, intermediate, hidden],
            ));
            tensors.push(expected_vector_shape(
                format!("{gguf}.ffn_{projection}_exps.bias"),
                vec![experts, intermediate],
            ));
        }
        tensors.push(expected_mxfp4_rank3(
            format!("{gguf}.ffn_down_exps.weight"),
            [experts, hidden, intermediate],
        ));
        tensors.push(expected_vector_shape(
            format!("{gguf}.ffn_down_exps.bias"),
            vec![experts, hidden],
        ));
    }
    tensors
}

fn kimi_linear_expected(args: &kimi_linear::ModelArgs) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        let prefix = format!("model.layers.{layer}");
        tensors.extend([
            expected_vector(
                format!("{prefix}.input_layernorm.weight"),
                format!("blk.{layer}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{prefix}.post_attention_layernorm.weight"),
                format!("blk.{layer}.ffn_norm.weight"),
                hidden,
            ),
        ]);
        if policy.attention == kimi_linear::AttentionKind::Kda {
            let heads = args.kda_config.num_heads as usize;
            let head = args.kda_config.head_dim as usize;
            let projection = heads * head;
            tensors.extend([
                expected(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    format!("blk.{layer}.attn_q.weight"),
                    [projection, hidden],
                ),
                expected(
                    format!("{prefix}.self_attn.k_proj.weight"),
                    format!("blk.{layer}.attn_k.weight"),
                    [projection, hidden],
                ),
                expected(
                    format!("{prefix}.self_attn.v_proj.weight"),
                    format!("blk.{layer}.attn_v.weight"),
                    [projection, hidden],
                ),
                expected(
                    format!("{prefix}.self_attn.f_a_proj.weight"),
                    format!("blk.{layer}.ssm_f_a.weight"),
                    [head, hidden],
                ),
                expected(
                    format!("{prefix}.self_attn.f_b_proj.weight"),
                    format!("blk.{layer}.ssm_f_b.weight"),
                    [projection, head],
                ),
                expected(
                    format!("{prefix}.self_attn.b_proj.weight"),
                    format!("blk.{layer}.ssm_beta.weight"),
                    [heads, hidden],
                ),
                expected(
                    format!("{prefix}.self_attn.g_a_proj.weight"),
                    format!("blk.{layer}.ssm_g_a.weight"),
                    [head, hidden],
                ),
                expected(
                    format!("{prefix}.self_attn.g_b_proj.weight"),
                    format!("blk.{layer}.ssm_g_b.weight"),
                    [projection, head],
                ),
                expected_vector(
                    format!("{prefix}.self_attn.dt_bias"),
                    format!("blk.{layer}.ssm_dt.bias"),
                    projection,
                ),
                expected_vector(
                    format!("{prefix}.self_attn.o_norm.weight"),
                    format!("blk.{layer}.ssm_norm.weight"),
                    head,
                ),
                expected(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    format!("blk.{layer}.attn_output.weight"),
                    [hidden, projection],
                ),
            ]);
        } else {
            let heads = args.num_attention_heads as usize;
            let query_head = (args.qk_nope_head_dim + args.qk_rope_head_dim) as usize;
            if let Some(rank) = args.q_lora_rank {
                tensors.extend([
                    expected(
                        format!("{prefix}.self_attn.q_a_proj.weight"),
                        format!("blk.{layer}.attn_q_a.weight"),
                        [rank as usize, hidden],
                    ),
                    expected_vector(
                        format!("{prefix}.self_attn.q_a_layernorm.weight"),
                        format!("blk.{layer}.attn_q_a_norm.weight"),
                        rank as usize,
                    ),
                    expected(
                        format!("{prefix}.self_attn.q_b_proj.weight"),
                        format!("blk.{layer}.attn_q_b.weight"),
                        [heads * query_head, rank as usize],
                    ),
                ]);
            } else {
                tensors.push(expected(
                    format!("{prefix}.self_attn.q_proj.weight"),
                    format!("blk.{layer}.attn_q.weight"),
                    [heads * query_head, hidden],
                ));
            }
            tensors.extend([
                expected(
                    format!("{prefix}.self_attn.kv_a_proj_with_mqa.weight"),
                    format!("blk.{layer}.attn_kv_a_mqa.weight"),
                    [(args.kv_lora_rank + args.qk_rope_head_dim) as usize, hidden],
                ),
                expected_vector(
                    format!("{prefix}.self_attn.kv_a_layernorm.weight"),
                    format!("blk.{layer}.attn_kv_a_norm.weight"),
                    args.kv_lora_rank as usize,
                ),
                expected(
                    format!("{prefix}.self_attn.o_proj.weight"),
                    format!("blk.{layer}.attn_output.weight"),
                    [hidden, heads * args.v_head_dim as usize],
                ),
            ]);
            if args.split_kv_b {
                tensors.extend([
                    expected(
                        format!("{prefix}.self_attn.k_b_proj.weight"),
                        format!("blk.{layer}.attn_k_b.weight"),
                        [
                            heads * args.qk_nope_head_dim as usize,
                            args.kv_lora_rank as usize,
                        ],
                    ),
                    expected(
                        format!("{prefix}.self_attn.v_b_proj.weight"),
                        format!("blk.{layer}.attn_v_b.weight"),
                        [heads * args.v_head_dim as usize, args.kv_lora_rank as usize],
                    ),
                ]);
            } else {
                tensors.push(expected(
                    format!("{prefix}.self_attn.kv_b_proj.weight"),
                    format!("blk.{layer}.attn_kv_b.weight"),
                    [
                        heads * (args.qk_nope_head_dim + args.v_head_dim) as usize,
                        args.kv_lora_rank as usize,
                    ],
                ));
            }
        }
        if policy.feed_forward == kimi_linear::FeedForwardPolicy::SparseMoe {
            let experts = args.num_experts as usize;
            let intermediate = args.moe_intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{prefix}.mlp.gate.weight"),
                    format!("blk.{layer}.ffn_gate_inp.weight"),
                    [experts, hidden],
                ),
                expected_vector(
                    format!("{prefix}.mlp.gate.e_score_correction_bias"),
                    format!("blk.{layer}.exp_probs_b.bias"),
                    experts,
                ),
                expected(
                    format!("{prefix}.mlp.shared_experts.gate_proj.weight"),
                    format!("blk.{layer}.ffn_gate_shexp.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.shared_experts.up_proj.weight"),
                    format!("blk.{layer}.ffn_up_shexp.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.shared_experts.down_proj.weight"),
                    format!("blk.{layer}.ffn_down_shexp.weight"),
                    [hidden, intermediate],
                ),
            ]);
        } else {
            let intermediate = args.intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{prefix}.mlp.gate_proj.weight"),
                    format!("blk.{layer}.ffn_gate.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.up_proj.weight"),
                    format!("blk.{layer}.ffn_up.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{prefix}.mlp.down_proj.weight"),
                    format!("blk.{layer}.ffn_down.weight"),
                    [hidden, intermediate],
                ),
            ]);
        }
    }
    tensors
}

fn lfm2_expected(args: &lfm2::ModelArgs) -> Result<Vec<ExpectedTensor>, Error> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let head = (args.hidden_size / args.num_attention_heads) as usize;
    let key_value =
        (args.num_key_value_heads * args.hidden_size / args.num_attention_heads) as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector(
            "model.embedding_norm.weight",
            "token_embd_norm.weight",
            hidden,
        ),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.operator_norm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.ffn_norm.weight"),
                format!("{gguf}.ffn_norm.weight"),
                hidden,
            ),
        ]);
        let policy = args.layer_schedule.get(layer).ok_or_else(|| {
            Error::UnsupportedArchitecture(format!(
                "LFM2 layer schedule has no policy for layer {layer}"
            ))
        })?;
        match policy.operator {
            lfm2::OperatorPolicy::CausalConvolution => {
                tensors.extend([
                    expected_dense_with_gguf_shape(
                        format!("{model}.conv.conv.weight"),
                        format!("{gguf}.shortconv.conv.weight"),
                        vec![hidden, 1, args.conv_l_cache as usize],
                        vec![hidden, args.conv_l_cache as usize],
                    ),
                    expected(
                        format!("{model}.conv.in_proj.weight"),
                        format!("{gguf}.shortconv.in_proj.weight"),
                        [3 * hidden, hidden],
                    ),
                    expected(
                        format!("{model}.conv.out_proj.weight"),
                        format!("{gguf}.shortconv.out_proj.weight"),
                        [hidden, hidden],
                    ),
                ]);
                if args.conv_bias {
                    tensors.extend([
                        expected_vector(
                            format!("{model}.conv.conv.bias"),
                            format!("{gguf}.shortconv.conv.bias"),
                            hidden,
                        ),
                        expected_vector(
                            format!("{model}.conv.in_proj.bias"),
                            format!("{gguf}.shortconv.in_proj.bias"),
                            3 * hidden,
                        ),
                        expected_vector(
                            format!("{model}.conv.out_proj.bias"),
                            format!("{gguf}.shortconv.out_proj.bias"),
                            hidden,
                        ),
                    ]);
                }
            }
            lfm2::OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full) => tensors.extend([
                expected(
                    format!("{model}.self_attn.q_proj.weight"),
                    format!("{gguf}.attn_q.weight"),
                    [hidden, hidden],
                ),
                expected(
                    format!("{model}.self_attn.k_proj.weight"),
                    format!("{gguf}.attn_k.weight"),
                    [key_value, hidden],
                ),
                expected(
                    format!("{model}.self_attn.v_proj.weight"),
                    format!("{gguf}.attn_v.weight"),
                    [key_value, hidden],
                ),
                expected(
                    format!("{model}.self_attn.out_proj.weight"),
                    format!("{gguf}.attn_output.weight"),
                    [hidden, hidden],
                ),
                expected_vector(
                    format!("{model}.self_attn.q_layernorm.weight"),
                    format!("{gguf}.attn_q_norm.weight"),
                    head,
                ),
                expected_vector(
                    format!("{model}.self_attn.k_layernorm.weight"),
                    format!("{gguf}.attn_k_norm.weight"),
                    head,
                ),
            ]),
            lfm2::OperatorPolicy::SelfAttention(crate::AttentionPolicy::Sliding { .. }) => {
                return Err(Error::UnsupportedArchitecture(
                    "LFM2 structural admission does not support sliding attention".into(),
                ));
            }
        }
        if policy.feed_forward == lfm2::FeedForwardPolicy::SparseMoe {
            let experts = args.num_experts as usize;
            let intermediate = args.moe_intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{model}.feed_forward.gate.weight"),
                    format!("{gguf}.ffn_gate_inp.weight"),
                    [experts, hidden],
                ),
                expected_rank3(
                    format!("{model}.feed_forward.experts.gate_proj"),
                    format!("{gguf}.ffn_gate_exps.weight"),
                    [experts, intermediate, hidden],
                ),
                expected_rank3(
                    format!("{model}.feed_forward.experts.up_proj"),
                    format!("{gguf}.ffn_up_exps.weight"),
                    [experts, intermediate, hidden],
                ),
                expected_rank3(
                    format!("{model}.feed_forward.experts.down_proj"),
                    format!("{gguf}.ffn_down_exps.weight"),
                    [experts, hidden, intermediate],
                ),
            ]);
            if args.use_expert_bias {
                tensors.push(expected_vector(
                    format!("{model}.feed_forward.expert_bias"),
                    format!("{gguf}.ffn_exp_probs_b.bias"),
                    experts,
                ));
            }
        } else {
            let intermediate = args.dense_intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{model}.feed_forward.w1.weight"),
                    format!("{gguf}.ffn_gate.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{model}.feed_forward.w2.weight"),
                    format!("{gguf}.ffn_down.weight"),
                    [hidden, intermediate],
                ),
                expected(
                    format!("{model}.feed_forward.w3.weight"),
                    format!("{gguf}.ffn_up.weight"),
                    [intermediate, hidden],
                ),
            ]);
        }
    }
    Ok(tensors)
}

fn expected(
    safetensors_name: impl Into<String>,
    gguf_name: impl Into<String>,
    shape: [usize; 2],
) -> ExpectedTensor {
    let shape = shape.to_vec();
    ExpectedTensor {
        safetensors_name: safetensors_name.into(),
        gguf_name: gguf_name.into(),
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Matrix,
    }
}

fn expected_vector(
    safetensors_name: impl Into<String>,
    gguf_name: impl Into<String>,
    size: usize,
) -> ExpectedTensor {
    let shape = vec![size];
    ExpectedTensor {
        safetensors_name: safetensors_name.into(),
        gguf_name: gguf_name.into(),
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Vector,
    }
}

fn expected_rank3(
    safetensors_name: impl Into<String>,
    gguf_name: impl Into<String>,
    shape: [usize; 3],
) -> ExpectedTensor {
    let shape = shape.to_vec();
    ExpectedTensor {
        safetensors_name: safetensors_name.into(),
        gguf_name: gguf_name.into(),
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Matrix,
    }
}

fn expected_mxfp4_rank3(gguf_name: impl Into<String>, shape: [usize; 3]) -> ExpectedTensor {
    let name = gguf_name.into();
    let shape = shape.to_vec();
    ExpectedTensor {
        safetensors_name: name.clone(),
        gguf_name: name,
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::MxFp4Matrix,
    }
}

fn expected_vector_shape(gguf_name: impl Into<String>, shape: Vec<usize>) -> ExpectedTensor {
    let name = gguf_name.into();
    ExpectedTensor {
        safetensors_name: name.clone(),
        gguf_name: name,
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Vector,
    }
}

fn expected_dense_with_gguf_shape(
    safetensors_name: impl Into<String>,
    gguf_name: impl Into<String>,
    safetensors_shape: impl Into<Vec<usize>>,
    gguf_shape: impl Into<Vec<usize>>,
) -> ExpectedTensor {
    ExpectedTensor {
        safetensors_name: safetensors_name.into(),
        gguf_name: gguf_name.into(),
        safetensors_shape: safetensors_shape.into(),
        gguf_shape: gguf_shape.into(),
        operation: TensorOperation::Dense,
    }
}

fn validate_inkling_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match inkling::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let text = &args.text_config;
    let hidden = text.hidden_size as usize;
    let vocab = text.vocab_size as usize;
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut allowed = BTreeSet::new();
    let mut issues = Vec::new();

    for (official, canonical, shape) in [
        (
            "model.llm.embed.weight".into(),
            "model.embed_tokens.weight".into(),
            vec![vocab, hidden],
        ),
        (
            "model.llm.embed_norm.weight".into(),
            "model.embed_norm.weight".into(),
            vec![hidden],
        ),
        (
            "model.llm.norm.weight".into(),
            "model.norm.weight".into(),
            vec![hidden],
        ),
        (
            "model.llm.unembed.weight".into(),
            "lm_head.weight".into(),
            vec![vocab, hidden],
        ),
    ] {
        validate_inkling_alias(
            store,
            &keys,
            &mut allowed,
            &mut issues,
            official,
            canonical,
            shape,
        );
    }

    for layer in 0..text.num_hidden_layers {
        let official = format!("model.llm.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        let policy = *text
            .layer_policy(layer as usize)
            .expect("validated Inkling layer schedule");
        let local = policy.attention.window().is_some();
        let query_heads = text.q_heads(local) as usize;
        let key_value_heads = text.kv_heads(local) as usize;
        let head = text.attention_head_dim(local) as usize;
        let relative = policy
            .attention
            .window()
            .map(|window| window.get() as usize)
            .unwrap_or(text.rel_extent as usize);
        for (source, target, shape) in [
            (
                format!("{official}.attn_norm.weight"),
                format!("{canonical}.input_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{official}.mlp_norm.weight"),
                format!("{canonical}.post_attention_layernorm.weight"),
                vec![hidden],
            ),
            (
                format!("{official}.attn.wq_du.weight"),
                format!("{canonical}.self_attn.q_proj.weight"),
                vec![query_heads * head, hidden],
            ),
            (
                format!("{official}.attn.wk_dv.weight"),
                format!("{canonical}.self_attn.k_proj.weight"),
                vec![key_value_heads * head, hidden],
            ),
            (
                format!("{official}.attn.wv_dv.weight"),
                format!("{canonical}.self_attn.v_proj.weight"),
                vec![key_value_heads * head, hidden],
            ),
            (
                format!("{official}.attn.wr_du.weight"),
                format!("{canonical}.self_attn.r_proj.weight"),
                vec![query_heads * text.d_rel as usize, hidden],
            ),
            (
                format!("{official}.attn.wo_ud.weight"),
                format!("{canonical}.self_attn.o_proj.weight"),
                vec![hidden, query_heads * head],
            ),
            (
                format!("{official}.attn.q_norm.weight"),
                format!("{canonical}.self_attn.q_norm.weight"),
                vec![head],
            ),
            (
                format!("{official}.attn.k_norm.weight"),
                format!("{canonical}.self_attn.k_norm.weight"),
                vec![head],
            ),
            (
                format!("{official}.attn.rel_logits_proj.proj"),
                format!("{canonical}.self_attn.rel_proj"),
                vec![text.d_rel as usize, relative],
            ),
            (
                format!("{official}.attn.k_sconv.weight"),
                format!("{canonical}.self_attn.k_sconv.weight"),
                vec![key_value_heads * head, 1, text.sconv_kernel_size as usize],
            ),
            (
                format!("{official}.attn.v_sconv.weight"),
                format!("{canonical}.self_attn.v_sconv.weight"),
                vec![key_value_heads * head, 1, text.sconv_kernel_size as usize],
            ),
            (
                format!("{official}.attn_sconv.weight"),
                format!("{canonical}.attn_sconv.weight"),
                vec![hidden, 1, text.sconv_kernel_size as usize],
            ),
            (
                format!("{official}.mlp_sconv.weight"),
                format!("{canonical}.mlp_sconv.weight"),
                vec![hidden, 1, text.sconv_kernel_size as usize],
            ),
        ] {
            validate_inkling_alias(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                source,
                target,
                shape,
            );
        }

        if policy.feed_forward == inkling::FeedForwardPolicy::Dense {
            let intermediate = text.dense_intermediate_size() as usize;
            validate_inkling_dense_w13(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                format!("{official}.mlp.w13_dn.weight"),
                format!("{canonical}.dense.gate_proj.weight"),
                format!("{canonical}.dense.up_proj.weight"),
                intermediate,
                hidden,
            );
            for (source, target, shape) in [
                (
                    format!("{official}.mlp.w2_md.weight"),
                    format!("{canonical}.dense.down_proj.weight"),
                    vec![hidden, intermediate],
                ),
                (
                    format!("{official}.mlp.global_scale"),
                    format!("{canonical}.dense_global_scale"),
                    vec![1],
                ),
            ] {
                validate_inkling_alias(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    source,
                    target,
                    shape,
                );
            }
        } else {
            let routed = text.n_routed_experts as usize;
            let shared = text.n_shared_experts as usize;
            let intermediate = text.moe_intermediate_size() as usize;
            for (source, target, shape) in [
                (
                    format!("{official}.mlp.gate.weight"),
                    format!("{canonical}.moe.router.weight"),
                    vec![routed + shared, hidden],
                ),
                (
                    format!("{official}.mlp.gate.bias"),
                    format!("{canonical}.moe.router.bias"),
                    vec![routed],
                ),
                (
                    format!("{official}.mlp.gate.global_scale"),
                    format!("{canonical}.moe.router.global_scale"),
                    vec![1],
                ),
                (
                    format!("{official}.mlp.experts.w13_weight"),
                    format!("{canonical}.moe.experts.gate_up_proj"),
                    vec![routed, intermediate * 2, hidden],
                ),
                (
                    format!("{official}.mlp.experts.w2_weight"),
                    format!("{canonical}.moe.experts.down_proj"),
                    vec![routed, hidden, intermediate],
                ),
                (
                    format!("{official}.mlp.shared_experts.shared_w13_weight"),
                    format!("{canonical}.moe.shared_experts.gate_up_proj"),
                    vec![shared, intermediate * 2, hidden],
                ),
                (
                    format!("{official}.mlp.shared_experts.shared_w2_weight"),
                    format!("{canonical}.moe.shared_experts.down_proj"),
                    vec![shared, hidden, intermediate],
                ),
            ] {
                validate_inkling_alias(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    source,
                    target,
                    shape,
                );
            }
        }
    }

    if let Some(audio) = &args.audio_config {
        for (source, target, shape) in [
            (
                "model.audio.encoder.weight".into(),
                "audio.encoder.weight".into(),
                vec![(audio.num_codebooks * audio.codebook_size) as usize, hidden],
            ),
            (
                "model.audio.final_norm.weight".into(),
                "audio.final_norm.weight".into(),
                vec![hidden],
            ),
        ] {
            validate_inkling_alias(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                source,
                target,
                shape,
            );
        }
    }
    if let Some(vision) = &args.vision_config {
        for (layer, (input, output, _, _)) in vision.layer_specs().into_iter().enumerate() {
            validate_inkling_alias(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                format!("model.visual.layers.linear_{layer}.weight"),
                format!("visual.layers.{layer}.projection.weight"),
                vec![output as usize, input as usize],
            );
            if layer + 1 != vision.layer_specs().len() {
                validate_inkling_alias(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    format!("model.visual.layers.norm_{layer}.weight"),
                    format!("visual.layers.{layer}.layer_norm.weight"),
                    vec![output as usize],
                );
            }
        }
        validate_inkling_alias(
            store,
            &keys,
            &mut allowed,
            &mut issues,
            "model.visual.final_norm.weight".into(),
            "visual.final_norm.weight".into(),
            vec![hidden],
        );
    }

    if let Some(mtp) = &args.mtp_config {
        let count = mtp.num_nextn_predict_layers as usize;
        for depth in 0..count {
            for (suffix, shape) in [
                ("hidden_norm.weight", vec![hidden]),
                ("embed_norm.weight", vec![hidden]),
                ("input_proj.weight", vec![hidden, hidden * 2]),
            ] {
                let key = format!("model.mtp.layers.{depth}.{suffix}");
                allowed.insert(key.clone());
                validate_safetensor(store, &key, &shape, false, &mut issues);
            }
            let local = mtp.local_layer_ids.contains(&depth);
            let query_heads = if local {
                mtp.swa_num_attention_heads
                    .or(text.swa_num_attention_heads)
                    .unwrap_or(mtp.num_attention_heads.unwrap_or(text.num_attention_heads))
            } else {
                mtp.num_attention_heads.unwrap_or(text.num_attention_heads)
            } as usize;
            let key_value_heads = if local {
                mtp.swa_num_key_value_heads
                    .or(text.swa_num_key_value_heads)
                    .unwrap_or(mtp.num_key_value_heads.unwrap_or(text.num_key_value_heads))
            } else {
                mtp.num_key_value_heads.unwrap_or(text.num_key_value_heads)
            } as usize;
            let head = if local {
                mtp.swa_head_dim
                    .or(text.swa_head_dim)
                    .unwrap_or(mtp.head_dim.unwrap_or(text.head_dim))
            } else {
                mtp.head_dim.unwrap_or(text.head_dim)
            } as usize;
            let d_rel = mtp.d_rel.unwrap_or(text.d_rel) as usize;
            let relative = if local {
                text.layer_schedule
                    .iter()
                    .find_map(|policy| policy.attention.window())
                    .map_or(
                        mtp.rel_extent.unwrap_or(text.rel_extent) as usize,
                        |window| window.get() as usize,
                    )
            } else {
                mtp.rel_extent.unwrap_or(text.rel_extent) as usize
            };
            let convolution = mtp.sconv_kernel_size.unwrap_or(text.sconv_kernel_size) as usize;
            let intermediate = mtp
                .dense_intermediate_size
                .or(text.dense_intermediate_size)
                .unwrap_or(mtp.intermediate_size.unwrap_or(text.intermediate_size))
                as usize;
            let prefix = format!("model.mtp.layers.{depth}.transformer_block");
            for (suffix, shape) in [
                ("attn_norm.weight", vec![hidden]),
                ("mlp_norm.weight", vec![hidden]),
                ("attn.wq_du.weight", vec![query_heads * head, hidden]),
                ("attn.wk_dv.weight", vec![key_value_heads * head, hidden]),
                ("attn.wv_dv.weight", vec![key_value_heads * head, hidden]),
                ("attn.wr_du.weight", vec![query_heads * d_rel, hidden]),
                ("attn.wo_ud.weight", vec![hidden, query_heads * head]),
                ("attn.q_norm.weight", vec![head]),
                ("attn.k_norm.weight", vec![head]),
                ("attn.rel_logits_proj.proj", vec![d_rel, relative]),
                (
                    "attn.k_sconv.weight",
                    vec![key_value_heads * head, 1, convolution],
                ),
                (
                    "attn.v_sconv.weight",
                    vec![key_value_heads * head, 1, convolution],
                ),
                ("attn_sconv.weight", vec![hidden, 1, convolution]),
                ("mlp_sconv.weight", vec![hidden, 1, convolution]),
                ("mlp.w13_dn.weight", vec![intermediate * 2, hidden]),
                ("mlp.w2_md.weight", vec![hidden, intermediate]),
                ("mlp.global_scale", vec![1]),
            ] {
                let key = format!("{prefix}.{suffix}");
                allowed.insert(key.clone());
                validate_safetensor(store, &key, &shape, false, &mut issues);
            }
        }
        if mtp.chain_hidden_post_norm {
            let key = "model.mtp.chain_norm.weight".to_string();
            allowed.insert(key.clone());
            validate_safetensor(store, &key, &[hidden], false, &mut issues);
        }
    }
    for key in keys {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(&key, "Inkling SafeTensors"));
        }
    }
    finish(issues)
}

#[allow(clippy::too_many_arguments)]
fn validate_inkling_alias(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    official: String,
    canonical: String,
    shape: Vec<usize>,
) {
    allowed.insert(official.clone());
    allowed.insert(canonical.clone());
    let present = [official.clone(), canonical]
        .into_iter()
        .filter(|name| keys.contains(name))
        .collect::<Vec<_>>();
    match present.as_slice() {
        [] => issues.push(missing(&official)),
        [name] => validate_safetensor(store, name, &shape, false, issues),
        [_, conflicting] => issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "Inkling checkpoint contains both released and canonical aliases for {official:?}"
            ),
            tensor_name: Some(conflicting.clone()),
            tensor_type_code: None,
            metadata_key: None,
        }),
        _ => unreachable!("two Inkling aliases"),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_inkling_dense_w13(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    official: String,
    canonical_gate: String,
    canonical_up: String,
    intermediate: usize,
    hidden: usize,
) {
    allowed.extend([
        official.clone(),
        canonical_gate.clone(),
        canonical_up.clone(),
    ]);
    let has_official = keys.contains(&official);
    let has_canonical = keys.contains(&canonical_gate) || keys.contains(&canonical_up);
    if has_official && has_canonical {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "Inkling dense layer mixes released interleaved tensor {official:?} with canonical gate/up tensors"
            ),
            tensor_name: Some(official),
            tensor_type_code: None,
            metadata_key: None,
        });
    } else if has_official {
        validate_safetensor(store, &official, &[intermediate * 2, hidden], false, issues);
    } else {
        validate_safetensor(
            store,
            &canonical_gate,
            &[intermediate, hidden],
            false,
            issues,
        );
        validate_safetensor(store, &canonical_up, &[intermediate, hidden], false, issues);
    }
}

fn validate_gemma4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let (args, vision, _, _, audio, _) = match gemma4::model_config_from_value(config) {
        Ok(config) => config,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if options.weight_residency.expert_cache().is_some() {
        return invalid_geometry(
            "Gemma 4 SafeTensors does not expose a sparse-expert-cache load route".into(),
        );
    }

    let hidden = args.hidden_size as usize;
    let layers = args.num_hidden_layers as usize;
    let vocab = args.vocab_size as usize;
    let quantization = args.weight_quantization();
    let bounded = !options.weight_residency.is_fully_resident();
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut allowed = BTreeSet::new();
    let mut issues = Vec::new();

    validate_gemma4_tensor(
        store,
        &keys,
        &mut allowed,
        &mut issues,
        "model.language_model.embed_tokens.weight".into(),
        vec![vocab, hidden],
        TensorOperation::Matrix,
        quantization,
        true,
    );
    validate_gemma4_tensor(
        store,
        &keys,
        &mut allowed,
        &mut issues,
        "model.language_model.norm.weight".into(),
        vec![hidden],
        TensorOperation::Vector,
        quantization,
        true,
    );
    if !args.tie_word_embeddings {
        validate_gemma4_tensor(
            store,
            &keys,
            &mut allowed,
            &mut issues,
            "lm_head.weight".into(),
            vec![vocab, hidden],
            TensorOperation::Matrix,
            quantization,
            false,
        );
    }
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = args.hidden_size_per_layer_input as usize;
        let combined = layers * per_layer;
        let per_layer_vocab = args.vocab_size_per_layer_input.unwrap_or(args.vocab_size) as usize;
        for (name, shape, operation) in [
            (
                "model.language_model.embed_tokens_per_layer.weight",
                vec![per_layer_vocab, combined],
                TensorOperation::Matrix,
            ),
            (
                "model.language_model.per_layer_model_projection.weight",
                vec![combined, hidden],
                TensorOperation::Matrix,
            ),
            (
                "model.language_model.per_layer_projection_norm.weight",
                vec![per_layer],
                TensorOperation::Vector,
            ),
        ] {
            validate_gemma4_tensor(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                name.into(),
                shape,
                operation,
                quantization,
                true,
            );
        }
    }

    for layer in 0..layers {
        let prefix = format!("model.language_model.layers.{layer}");
        let policy = args
            .layer_policy(layer)
            .expect("validated Gemma 4 layer schedule");
        let head_dim = policy.head_dim.get() as usize;
        let kv_heads = policy.num_key_value_heads.get() as usize;
        let query = args.num_attention_heads as usize * head_dim;
        let key_value = kv_heads * head_dim;
        let shared_kv = !policy.key_value.owns_state();
        let attention_k_eq_v = policy.key_value.value() == Some(gemma4::ValuePolicy::ReuseKey);
        let intermediate = policy.intermediate_size.get() as usize;

        for name in [
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ] {
            validate_gemma4_tensor(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                format!("{prefix}.{name}.weight"),
                vec![hidden],
                TensorOperation::Vector,
                quantization,
                true,
            );
        }
        validate_gemma4_tensor(
            store,
            &keys,
            &mut allowed,
            &mut issues,
            format!("{prefix}.layer_scalar"),
            vec![1],
            TensorOperation::Vector,
            quantization,
            true,
        );
        for (name, shape, operation) in [
            (
                "self_attn.q_proj.weight",
                vec![query, hidden],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.o_proj.weight",
                vec![hidden, query],
                TensorOperation::Matrix,
            ),
            (
                "self_attn.q_norm.weight",
                vec![head_dim],
                TensorOperation::Vector,
            ),
            (
                "mlp.gate_proj.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.up_proj.weight",
                vec![intermediate, hidden],
                TensorOperation::Matrix,
            ),
            (
                "mlp.down_proj.weight",
                vec![hidden, intermediate],
                TensorOperation::Matrix,
            ),
        ] {
            validate_gemma4_tensor(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                format!("{prefix}.{name}"),
                shape,
                operation,
                quantization,
                true,
            );
        }
        if !shared_kv {
            for (name, shape, operation) in [
                (
                    "self_attn.k_proj.weight",
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "self_attn.k_norm.weight",
                    vec![head_dim],
                    TensorOperation::Vector,
                ),
            ] {
                validate_gemma4_tensor(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    format!("{prefix}.{name}"),
                    shape,
                    operation,
                    quantization,
                    true,
                );
            }
            if !attention_k_eq_v {
                validate_gemma4_tensor(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    format!("{prefix}.self_attn.v_proj.weight"),
                    vec![key_value, hidden],
                    TensorOperation::Matrix,
                    quantization,
                    true,
                );
            }
        }
        if args.hidden_size_per_layer_input > 0 {
            let per_layer = args.hidden_size_per_layer_input as usize;
            for (name, shape, operation) in [
                (
                    "per_layer_input_gate.weight",
                    vec![per_layer, hidden],
                    TensorOperation::Matrix,
                ),
                (
                    "per_layer_projection.weight",
                    vec![hidden, per_layer],
                    TensorOperation::Matrix,
                ),
                (
                    "post_per_layer_input_norm.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
            ] {
                validate_gemma4_tensor(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    format!("{prefix}.{name}"),
                    shape,
                    operation,
                    quantization,
                    true,
                );
            }
        }
        if policy.feed_forward == gemma4::FeedForwardPolicy::DenseWithSparseMoe {
            let experts = args.num_experts.expect("validated Gemma 4 MoE") as usize;
            let moe_intermediate =
                args.moe_intermediate_size.expect("validated Gemma 4 MoE") as usize;
            for (name, shape, operation) in [
                (
                    "router.proj.weight",
                    vec![experts, hidden],
                    TensorOperation::Matrix,
                ),
                ("router.scale", vec![hidden], TensorOperation::Vector),
                (
                    "router.per_expert_scale",
                    vec![experts],
                    TensorOperation::Vector,
                ),
                (
                    "post_feedforward_layernorm_1.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                (
                    "pre_feedforward_layernorm_2.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
                (
                    "post_feedforward_layernorm_2.weight",
                    vec![hidden],
                    TensorOperation::Vector,
                ),
            ] {
                validate_gemma4_tensor(
                    store,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    format!("{prefix}.{name}"),
                    shape,
                    operation,
                    quantization,
                    true,
                );
            }
            validate_gemma4_experts(
                store,
                &keys,
                &mut allowed,
                &mut issues,
                &prefix,
                experts,
                hidden,
                moe_intermediate,
                quantization,
                bounded,
            );
        }
    }

    if let Some(vision) = vision.as_ref() {
        validate_gemma4_vision_catalog(
            store,
            &keys,
            &mut allowed,
            &mut issues,
            vision,
            hidden,
            quantization,
        );
    }
    if let Some(audio) = audio.as_ref() {
        validate_gemma4_audio_catalog(
            store,
            &keys,
            &mut allowed,
            &mut issues,
            audio,
            hidden,
            quantization,
        );
    }

    for key in keys {
        if allowed.contains(&key)
            || [
                "multi_modal_projector.",
                "model.multi_modal_projector.",
                "model.vision_embedder.",
            ]
            .iter()
            .any(|prefix| key.starts_with(prefix))
        {
            continue;
        }
        issues.push(unexpected_layout(&key, "Gemma 4 SafeTensors"));
    }
    finish(issues)
}

fn validate_gemma4_vision_catalog(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    config: &crate::architectures::gemma4::vision::Gemma4VisionConfig,
    text_hidden: usize,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
) {
    let hidden = config.hidden_size as usize;
    let intermediate = config.intermediate_size as usize;
    let query = config.num_attention_heads as usize * config.head_dim as usize;
    let key_value = config.num_key_value_heads as usize * config.head_dim as usize;
    let root = "model.vision_tower";
    for (name, shape) in [
        (
            format!("{root}.patch_embedder.input_proj.weight"),
            vec![hidden, 3 * (config.patch_size as usize).pow(2)],
        ),
        (
            format!("{root}.patch_embedder.position_embedding_table"),
            vec![2, config.position_embedding_size as usize, hidden],
        ),
    ] {
        validate_gemma4_media_tensor(
            store,
            keys,
            allowed,
            issues,
            name,
            shape,
            SafetensorsMatrixFormat::Dense,
        );
    }
    if config.standardize {
        for name in ["std_bias", "std_scale"] {
            validate_gemma4_media_tensor(
                store,
                keys,
                allowed,
                issues,
                format!("{root}.{name}"),
                vec![hidden],
                SafetensorsMatrixFormat::Dense,
            );
        }
    }
    for layer in 0..config.num_hidden_layers as usize {
        let prefix = format!("{root}.encoder.layers.{layer}");
        for name in [
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        ] {
            validate_gemma4_media_tensor(
                store,
                keys,
                allowed,
                issues,
                format!("{prefix}.{name}"),
                vec![hidden],
                SafetensorsMatrixFormat::Dense,
            );
        }
        for (name, shape) in [
            ("self_attn.q_proj", vec![query, hidden]),
            ("self_attn.k_proj", vec![key_value, hidden]),
            ("self_attn.v_proj", vec![key_value, hidden]),
            ("self_attn.o_proj", vec![hidden, query]),
            ("mlp.gate_proj", vec![intermediate, hidden]),
            ("mlp.up_proj", vec![intermediate, hidden]),
            ("mlp.down_proj", vec![hidden, intermediate]),
        ] {
            validate_gemma4_clipped_linear(
                store,
                keys,
                allowed,
                issues,
                &format!("{prefix}.{name}"),
                shape,
            );
        }
        for name in ["self_attn.q_norm.weight", "self_attn.k_norm.weight"] {
            validate_gemma4_media_tensor(
                store,
                keys,
                allowed,
                issues,
                format!("{prefix}.{name}"),
                vec![config.head_dim as usize],
                SafetensorsMatrixFormat::Dense,
            );
        }
    }
    validate_gemma4_media_tensor(
        store,
        keys,
        allowed,
        issues,
        "model.embed_vision.embedding_projection.weight".into(),
        vec![text_hidden, hidden],
        quantization.map_or(
            SafetensorsMatrixFormat::Dense,
            SafetensorsMatrixFormat::Affine,
        ),
    );
}

fn validate_gemma4_audio_catalog(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    config: &crate::architectures::gemma4::audio::Gemma4AudioConfig,
    text_hidden: usize,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
) {
    let hidden = config.hidden_size as usize;
    let head = hidden / config.num_attention_heads as usize;
    let [first, second] = config.subsampling_conv_channels.as_slice() else {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::InvalidGeometry,
            detail: "Gemma 4 audio requires exactly two subsampling convolution channels".into(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: Some("audio_config.subsampling_conv_channels".into()),
        });
        return;
    };
    let first = *first as usize;
    let second = *second as usize;
    let root = "model.audio_tower";
    for (name, shape) in [
        (
            format!("{root}.subsample_conv_projection.layer0.conv.weight"),
            vec![first, 3, 3, 1],
        ),
        (
            format!("{root}.subsample_conv_projection.layer0.norm.weight"),
            vec![first],
        ),
        (
            format!("{root}.subsample_conv_projection.layer1.conv.weight"),
            vec![second, 3, 3, first],
        ),
        (
            format!("{root}.subsample_conv_projection.layer1.norm.weight"),
            vec![second],
        ),
        (
            format!("{root}.subsample_conv_projection.input_proj_linear.weight"),
            vec![hidden, 32 * second],
        ),
        (
            format!("{root}.output_proj.weight"),
            vec![config.output_proj_dims as usize, hidden],
        ),
    ] {
        validate_gemma4_media_tensor(
            store,
            keys,
            allowed,
            issues,
            name,
            shape,
            SafetensorsMatrixFormat::Dense,
        );
    }
    validate_optional_gemma4_media_tensor(
        store,
        keys,
        allowed,
        issues,
        &format!("{root}.output_proj.bias"),
        &[config.output_proj_dims as usize],
    );

    for layer in 0..config.num_hidden_layers as usize {
        let prefix = format!("{root}.layers.{layer}");
        for name in [
            "feed_forward1.pre_layer_norm.weight",
            "feed_forward1.post_layer_norm.weight",
            "norm_pre_attn.weight",
            "norm_post_attn.weight",
            "lconv1d.pre_layer_norm.weight",
            "lconv1d.conv_norm.weight",
            "feed_forward2.pre_layer_norm.weight",
            "feed_forward2.post_layer_norm.weight",
            "norm_out.weight",
        ] {
            validate_gemma4_media_tensor(
                store,
                keys,
                allowed,
                issues,
                format!("{prefix}.{name}"),
                vec![hidden],
                SafetensorsMatrixFormat::Dense,
            );
        }
        for (name, shape) in [
            ("feed_forward1.ffw_layer_1", vec![4 * hidden, hidden]),
            ("feed_forward1.ffw_layer_2", vec![hidden, 4 * hidden]),
            ("self_attn.q_proj", vec![hidden, hidden]),
            ("self_attn.k_proj", vec![hidden, hidden]),
            ("self_attn.v_proj", vec![hidden, hidden]),
            ("self_attn.post", vec![hidden, hidden]),
            ("lconv1d.linear_start", vec![2 * hidden, hidden]),
            ("lconv1d.linear_end", vec![hidden, hidden]),
            ("feed_forward2.ffw_layer_1", vec![4 * hidden, hidden]),
            ("feed_forward2.ffw_layer_2", vec![hidden, 4 * hidden]),
        ] {
            validate_gemma4_clipped_linear(
                store,
                keys,
                allowed,
                issues,
                &format!("{prefix}.{name}"),
                shape,
            );
        }
        for (name, shape) in [
            ("self_attn.relative_k_proj.weight", vec![hidden, hidden]),
            ("self_attn.per_dim_scale", vec![head]),
            (
                "lconv1d.depthwise_conv1d.weight",
                vec![hidden, config.conv_kernel_size as usize, 1],
            ),
        ] {
            validate_gemma4_media_tensor(
                store,
                keys,
                allowed,
                issues,
                format!("{prefix}.{name}"),
                shape,
                SafetensorsMatrixFormat::Dense,
            );
        }
    }
    validate_gemma4_media_tensor(
        store,
        keys,
        allowed,
        issues,
        "model.embed_audio.embedding_projection.weight".into(),
        vec![text_hidden, config.output_proj_dims as usize],
        quantization.map_or(
            SafetensorsMatrixFormat::Dense,
            SafetensorsMatrixFormat::Affine,
        ),
    );
}

fn validate_gemma4_clipped_linear(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    prefix: &str,
    shape: Vec<usize>,
) {
    validate_gemma4_media_tensor(
        store,
        keys,
        allowed,
        issues,
        format!("{prefix}.linear.weight"),
        shape,
        SafetensorsMatrixFormat::Dense,
    );
    for suffix in ["input_min", "input_max", "output_min", "output_max"] {
        validate_gemma4_media_tensor(
            store,
            keys,
            allowed,
            issues,
            format!("{prefix}.{suffix}"),
            vec![],
            SafetensorsMatrixFormat::Dense,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gemma4_media_tensor(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    canonical: String,
    shape: Vec<usize>,
    format: SafetensorsMatrixFormat,
) {
    let released = canonical.strip_prefix("model.").map(str::to_owned);
    let aliases = std::iter::once(canonical.clone())
        .chain(released)
        .collect::<Vec<_>>();
    for alias in &aliases {
        allowed.insert(alias.clone());
        add_safetensors_format_companions(allowed, alias, format);
    }
    let present = aliases
        .iter()
        .filter(|name| keys.contains(*name))
        .collect::<Vec<_>>();
    match present.as_slice() {
        [] => issues.push(missing(&canonical)),
        [name] => validate_safetensor_format(store, name, &shape, format, issues),
        [_, conflicting, ..] => issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "Gemma 4 multimodal checkpoint contains multiple aliases for {canonical:?}: {present:?}"
            ),
            tensor_name: Some((**conflicting).clone()),
            tensor_type_code: None,
            metadata_key: None,
        }),
    }
}

fn validate_optional_gemma4_media_tensor(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    canonical: &str,
    shape: &[usize],
) {
    let released = canonical.strip_prefix("model.").map(str::to_owned);
    for alias in std::iter::once(canonical.to_owned()).chain(released) {
        allowed.insert(alias.clone());
        if keys.contains(&alias) {
            validate_safetensor(store, &alias, shape, false, issues);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_gemma4_tensor(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    canonical: String,
    shape: Vec<usize>,
    operation: TensorOperation,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    released_alias: bool,
) {
    let released = released_alias
        .then(|| {
            canonical
                .strip_prefix("model.language_model.")
                .map(|rest| format!("language_model.model.{rest}"))
        })
        .flatten();
    let mut candidates = released
        .into_iter()
        .chain(std::iter::once(canonical.clone()))
        .collect::<Vec<_>>();
    candidates.dedup();
    if operation == TensorOperation::Matrix && quantization.is_some() {
        let packed_aliases = candidates
            .iter()
            .filter_map(|name| quantized_weight_alias(name))
            .collect::<Vec<_>>();
        candidates.extend(packed_aliases);
        candidates.dedup();
    }
    allowed.extend(candidates.iter().cloned());
    if operation == TensorOperation::Matrix {
        if let Some(quantization) = quantization {
            for name in &candidates {
                add_safetensors_format_companions(
                    allowed,
                    name,
                    SafetensorsMatrixFormat::Affine(quantization),
                );
            }
        }
    }
    let present = candidates
        .iter()
        .filter(|name| keys.contains(*name))
        .collect::<Vec<_>>();
    if present.is_empty() {
        issues.push(missing(candidates.first().unwrap_or(&canonical)));
        return;
    }
    if present.len() > 1 {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "Gemma 4 checkpoint contains multiple aliases for {:?}: {:?}",
                canonical, present
            ),
            tensor_name: Some((*present[1]).clone()),
            tensor_type_code: None,
            metadata_key: None,
        });
        return;
    }
    let name = (*present[0]).clone();
    if operation == TensorOperation::Matrix {
        if let Some(quantization) = quantization {
            validate_quantized_safetensor(
                store,
                &ExpectedTensor {
                    safetensors_name: name,
                    gguf_name: String::new(),
                    safetensors_shape: shape.clone(),
                    gguf_shape: shape,
                    operation,
                },
                quantization,
                issues,
            );
            return;
        }
    }
    validate_safetensor(store, &name, &shape, false, issues);
}

#[allow(clippy::too_many_arguments)]
fn validate_gemma4_experts(
    store: &SafetensorsWeightStore,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    layer_prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    bounded: bool,
) {
    let expert_prefix = format!("{layer_prefix}.experts.switch_glu");
    let fused = format!("{expert_prefix}.gate_up_proj.weight");
    let separate = [
        format!("{expert_prefix}.gate_proj.weight"),
        format!("{expert_prefix}.up_proj.weight"),
    ];
    let has_fused = keys.contains(&fused);
    let has_separate = separate.iter().any(|name| {
        keys.contains(name)
            || name
                .strip_prefix("model.language_model.")
                .is_some_and(|rest| keys.contains(&format!("language_model.model.{rest}")))
    });
    if has_fused && has_separate {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: "Gemma 4 MoE layer mixes fused and separate gate/up expert tensors".into(),
            tensor_name: Some(fused.clone()),
            tensor_type_code: None,
            metadata_key: None,
        });
    } else if bounded && has_fused {
        validate_gemma4_tensor(
            store,
            keys,
            allowed,
            issues,
            fused,
            vec![experts, 2 * intermediate, hidden],
            TensorOperation::Matrix,
            quantization,
            false,
        );
    } else {
        for projection in ["gate_proj", "up_proj"] {
            validate_gemma4_tensor(
                store,
                keys,
                allowed,
                issues,
                format!("{expert_prefix}.{projection}.weight"),
                vec![experts, intermediate, hidden],
                TensorOperation::Matrix,
                quantization,
                true,
            );
        }
    }
    validate_gemma4_tensor(
        store,
        keys,
        allowed,
        issues,
        format!("{expert_prefix}.down_proj.weight"),
        vec![experts, hidden, intermediate],
        TensorOperation::Matrix,
        quantization,
        true,
    );
}

fn validate_nemotron_h_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match nemotron_h::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let mut allowed = BTreeSet::new();
    let mut issues = Vec::new();
    for (official, canonical, shape) in [
        (
            "backbone.embeddings.weight".into(),
            "model.embeddings.weight".into(),
            vec![vocab, hidden],
        ),
        (
            "backbone.norm_f.weight".into(),
            "model.norm_f.weight".into(),
            vec![hidden],
        ),
    ] {
        validate_nemotron_alias(
            store,
            &args,
            &keys,
            &mut allowed,
            &mut issues,
            official,
            canonical,
            shape,
        );
    }
    if !args.tie_word_embeddings {
        allowed.insert("lm_head.weight".into());
        validate_safetensor(
            store,
            "lm_head.weight",
            &[vocab, hidden],
            false,
            &mut issues,
        );
    }

    for (layer, block_type) in args.layer_schedule.iter().copied().enumerate() {
        let official = format!("backbone.layers.{layer}");
        let canonical = format!("model.layers.{layer}");
        validate_nemotron_alias(
            store,
            &args,
            &keys,
            &mut allowed,
            &mut issues,
            format!("{official}.norm.weight"),
            format!("{canonical}.norm.weight"),
            vec![hidden],
        );
        match block_type {
            nemotron_h::LayerPolicy::Mamba => {
                let intermediate = (args.mamba_num_heads * args.mamba_head_dim) as usize;
                let conv = intermediate + 2 * args.n_groups as usize * args.ssm_state_size as usize;
                let projection = intermediate + conv + args.mamba_num_heads as usize;
                for (source, target, shape) in [
                    (
                        format!("{official}.mixer.in_proj.weight"),
                        format!("{canonical}.mamba.in_proj.weight"),
                        vec![projection, hidden],
                    ),
                    (
                        format!("{official}.mixer.conv1d.weight"),
                        format!("{canonical}.mamba.conv1d.weight"),
                        vec![conv, 1, args.conv_kernel as usize],
                    ),
                    (
                        format!("{official}.mixer.dt_bias"),
                        format!("{canonical}.mamba.dt_bias"),
                        vec![args.mamba_num_heads as usize],
                    ),
                    (
                        format!("{official}.mixer.A_log"),
                        format!("{canonical}.mamba.A_log"),
                        vec![args.mamba_num_heads as usize],
                    ),
                    (
                        format!("{official}.mixer.D"),
                        format!("{canonical}.mamba.D"),
                        vec![args.mamba_num_heads as usize],
                    ),
                    (
                        format!("{official}.mixer.norm.weight"),
                        format!("{canonical}.mamba.norm.weight"),
                        vec![intermediate],
                    ),
                    (
                        format!("{official}.mixer.out_proj.weight"),
                        format!("{canonical}.mamba.out_proj.weight"),
                        vec![hidden, intermediate],
                    ),
                ] {
                    validate_nemotron_alias(
                        store,
                        &args,
                        &keys,
                        &mut allowed,
                        &mut issues,
                        source,
                        target,
                        shape,
                    );
                }
                if args.use_conv_bias {
                    validate_nemotron_alias(
                        store,
                        &args,
                        &keys,
                        &mut allowed,
                        &mut issues,
                        format!("{official}.mixer.conv1d.bias"),
                        format!("{canonical}.mamba.conv1d.bias"),
                        vec![conv],
                    );
                }
                if args.use_bias {
                    for (source, target, size) in [
                        (
                            format!("{official}.mixer.in_proj.bias"),
                            format!("{canonical}.mamba.in_proj.bias"),
                            projection,
                        ),
                        (
                            format!("{official}.mixer.out_proj.bias"),
                            format!("{canonical}.mamba.out_proj.bias"),
                            hidden,
                        ),
                    ] {
                        validate_nemotron_alias(
                            store,
                            &args,
                            &keys,
                            &mut allowed,
                            &mut issues,
                            source,
                            target,
                            vec![size],
                        );
                    }
                }
            }
            nemotron_h::LayerPolicy::SelfAttention(_) => {
                let query = (args.num_attention_heads * args.head_dim) as usize;
                let key_value = (args.num_key_value_heads * args.head_dim) as usize;
                for (projection, output, input) in [
                    ("q_proj", query, hidden),
                    ("k_proj", key_value, hidden),
                    ("v_proj", key_value, hidden),
                    ("o_proj", hidden, query),
                ] {
                    validate_nemotron_alias(
                        store,
                        &args,
                        &keys,
                        &mut allowed,
                        &mut issues,
                        format!("{official}.mixer.{projection}.weight"),
                        format!("{canonical}.attention.{projection}.weight"),
                        vec![output, input],
                    );
                    if args.attention_bias {
                        validate_nemotron_alias(
                            store,
                            &args,
                            &keys,
                            &mut allowed,
                            &mut issues,
                            format!("{official}.mixer.{projection}.bias"),
                            format!("{canonical}.attention.{projection}.bias"),
                            vec![output],
                        );
                    }
                }
            }
            nemotron_h::LayerPolicy::DenseMlp => {
                let intermediate = args.intermediate_size as usize;
                for (projection, shape) in [
                    ("up_proj", vec![intermediate, hidden]),
                    ("down_proj", vec![hidden, intermediate]),
                ] {
                    validate_nemotron_alias(
                        store,
                        &args,
                        &keys,
                        &mut allowed,
                        &mut issues,
                        format!("{official}.mixer.{projection}.weight"),
                        format!("{canonical}.mlp.{projection}.weight"),
                        shape,
                    );
                    if args.mlp_bias {
                        let size = if projection == "up_proj" {
                            intermediate
                        } else {
                            hidden
                        };
                        validate_nemotron_alias(
                            store,
                            &args,
                            &keys,
                            &mut allowed,
                            &mut issues,
                            format!("{official}.mixer.{projection}.bias"),
                            format!("{canonical}.mlp.{projection}.bias"),
                            vec![size],
                        );
                    }
                }
            }
            nemotron_h::LayerPolicy::SparseMoe => {
                let experts = args.n_routed_experts as usize;
                let intermediate = args.moe_intermediate_size as usize;
                for (source, target, shape) in [
                    (
                        format!("{official}.mixer.gate.weight"),
                        format!("{canonical}.moe.gate.weight"),
                        vec![experts, hidden],
                    ),
                    (
                        format!("{official}.mixer.gate.e_score_correction_bias"),
                        format!("{canonical}.moe.gate.e_score_correction_bias"),
                        vec![experts],
                    ),
                    (
                        format!("{official}.mixer.shared_experts.up_proj.weight"),
                        format!("{canonical}.moe.shared_experts.up_proj.weight"),
                        vec![args.moe_shared_expert_intermediate_size as usize, hidden],
                    ),
                    (
                        format!("{official}.mixer.shared_experts.down_proj.weight"),
                        format!("{canonical}.moe.shared_experts.down_proj.weight"),
                        vec![hidden, args.moe_shared_expert_intermediate_size as usize],
                    ),
                ] {
                    validate_nemotron_alias(
                        store,
                        &args,
                        &keys,
                        &mut allowed,
                        &mut issues,
                        source,
                        target,
                        shape,
                    );
                }
                if args.mlp_bias {
                    for (projection, size) in [
                        ("up_proj", args.moe_shared_expert_intermediate_size as usize),
                        ("down_proj", hidden),
                    ] {
                        validate_nemotron_alias(
                            store,
                            &args,
                            &keys,
                            &mut allowed,
                            &mut issues,
                            format!("{official}.mixer.shared_experts.{projection}.bias"),
                            format!("{canonical}.moe.shared_experts.{projection}.bias"),
                            vec![size],
                        );
                    }
                }
                validate_nemotron_experts(
                    store,
                    &args,
                    &keys,
                    &mut allowed,
                    &mut issues,
                    layer,
                    experts,
                    hidden,
                    intermediate,
                );
            }
        }
    }
    if args.num_nextn_predict_layers > 0 {
        let policies = match args.mtp_policies() {
            Ok(policies) => policies,
            Err(error) => return invalid_geometry(error.to_string()),
        };
        let pattern_len = policies.len() / args.num_nextn_predict_layers as usize;
        for (layer, policy) in policies.iter().copied().enumerate() {
            let official = format!("mtp.layers.{layer}");
            let canonical = format!("model.mtp.layers.{layer}");
            validate_nemotron_alias(
                store,
                &args,
                &keys,
                &mut allowed,
                &mut issues,
                format!("{official}.norm.weight"),
                format!("{canonical}.norm.weight"),
                vec![hidden],
            );
            match policy {
                nemotron_h::LayerPolicy::SelfAttention(_) => {
                    let query = (args.num_attention_heads * args.head_dim) as usize;
                    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
                    for (projection, output, input) in [
                        ("q_proj", query, hidden),
                        ("k_proj", key_value, hidden),
                        ("v_proj", key_value, hidden),
                        ("o_proj", hidden, query),
                    ] {
                        validate_nemotron_alias(
                            store,
                            &args,
                            &keys,
                            &mut allowed,
                            &mut issues,
                            format!("{official}.mixer.{projection}.weight"),
                            format!("{canonical}.mixer.{projection}.weight"),
                            vec![output, input],
                        );
                        if args.attention_bias {
                            validate_nemotron_alias(
                                store,
                                &args,
                                &keys,
                                &mut allowed,
                                &mut issues,
                                format!("{official}.mixer.{projection}.bias"),
                                format!("{canonical}.mixer.{projection}.bias"),
                                vec![output],
                            );
                        }
                    }
                }
                nemotron_h::LayerPolicy::SparseMoe => {
                    let experts = args.n_routed_experts as usize;
                    let intermediate = args.moe_intermediate_size as usize;
                    for (suffix, shape) in [
                        ("gate.weight", vec![experts, hidden]),
                        ("gate.e_score_correction_bias", vec![experts]),
                        (
                            "shared_experts.up_proj.weight",
                            vec![args.moe_shared_expert_intermediate_size as usize, hidden],
                        ),
                        (
                            "shared_experts.down_proj.weight",
                            vec![hidden, args.moe_shared_expert_intermediate_size as usize],
                        ),
                    ] {
                        validate_nemotron_alias(
                            store,
                            &args,
                            &keys,
                            &mut allowed,
                            &mut issues,
                            format!("{official}.mixer.{suffix}"),
                            format!("{canonical}.mixer.{suffix}"),
                            shape,
                        );
                    }
                    if args.mlp_bias {
                        for (projection, size) in [
                            ("up_proj", args.moe_shared_expert_intermediate_size as usize),
                            ("down_proj", hidden),
                        ] {
                            validate_nemotron_alias(
                                store,
                                &args,
                                &keys,
                                &mut allowed,
                                &mut issues,
                                format!("{official}.mixer.shared_experts.{projection}.bias"),
                                format!("{canonical}.mixer.shared_experts.{projection}.bias"),
                                vec![size],
                            );
                        }
                    }
                    validate_nemotron_experts_at(
                        store,
                        &args,
                        &keys,
                        &mut allowed,
                        &mut issues,
                        &format!("MTP physical layer {layer}"),
                        &format!("{official}.mixer.experts"),
                        &format!("{canonical}.mixer.experts"),
                        experts,
                        hidden,
                        intermediate,
                    );
                }
                _ => unreachable!("validated Nemotron-H MTP policy"),
            }
        }
        for step in 0..args.num_nextn_predict_layers as usize {
            let start = step * pattern_len;
            let end = start + pattern_len - 1;
            for (layer, suffix, shape) in [
                (start, "enorm.weight", vec![hidden]),
                (start, "hnorm.weight", vec![hidden]),
                (start, "eh_proj.weight", vec![hidden, hidden * 2]),
                (end, "final_layernorm.weight", vec![hidden]),
            ] {
                let candidates = [
                    format!("mtp.layers.{layer}.{suffix}"),
                    format!("model.mtp.layers.{layer}.{suffix}"),
                ];
                let present = candidates
                    .iter()
                    .find(|candidate| keys.contains(*candidate));
                match present {
                    Some(key) => {
                        allowed.insert(key.clone());
                        validate_safetensor(store, key, &shape, false, &mut issues);
                    }
                    None => issues.push(missing(&candidates[0])),
                }
            }
        }
    }
    for key in keys {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(&key, "Nemotron-H SafeTensors"));
        }
    }
    finish(issues)
}

#[allow(clippy::too_many_arguments)]
fn validate_nemotron_alias(
    store: &SafetensorsWeightStore,
    args: &nemotron_h::ModelArgs,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    official: String,
    canonical: String,
    shape: Vec<usize>,
) {
    let quantized = shape.len() >= 2
        && canonical.ends_with(".weight")
        && !canonical.contains(".conv1d.weight")
        && !canonical.ends_with(".moe.gate.weight")
        && args.weight_quantization_for(&canonical).is_some();
    let mut candidates = vec![official.clone(), canonical.clone()];
    if quantized {
        candidates.extend(
            candidates
                .clone()
                .iter()
                .filter_map(|name| quantized_weight_alias(name)),
        );
    }
    candidates.sort();
    candidates.dedup();
    allowed.extend(candidates.iter().cloned());
    let present = candidates
        .iter()
        .filter(|name| keys.contains(*name))
        .collect::<Vec<_>>();
    match present.as_slice() {
        [] => issues.push(missing(&official)),
        [actual] => {
            validate_nemotron_tensor(store, args, actual, &canonical, &shape, allowed, issues)
        }
        [_, conflicting, ..] => issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "Nemotron-H checkpoint contains conflicting aliases for {official:?}: {present:?}"
            ),
            tensor_name: Some((**conflicting).clone()),
            tensor_type_code: None,
            metadata_key: None,
        }),
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_nemotron_tensor(
    store: &SafetensorsWeightStore,
    args: &nemotron_h::ModelArgs,
    actual_name: &str,
    canonical_name: &str,
    shape: &[usize],
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
) {
    let quantizable = shape.len() >= 2
        && (canonical_name.ends_with(".weight")
            || canonical_name.ends_with(".experts.up_proj")
            || canonical_name.ends_with(".experts.down_proj"))
        && !canonical_name.contains(".conv1d.weight")
        && !canonical_name.ends_with(".moe.gate.weight");
    let Some(quantization) = quantizable
        .then(|| args.weight_quantization_for(canonical_name))
        .flatten()
    else {
        validate_safetensor(store, actual_name, shape, false, issues);
        return;
    };
    let tensor = ExpectedTensor {
        safetensors_name: actual_name.into(),
        gguf_name: String::new(),
        safetensors_shape: shape.to_vec(),
        gguf_shape: shape.to_vec(),
        operation: TensorOperation::Matrix,
    };
    add_safetensors_format_companions(
        allowed,
        actual_name,
        SafetensorsMatrixFormat::Affine(quantization),
    );
    validate_quantized_safetensor(store, &tensor, quantization, issues);
}

#[allow(clippy::too_many_arguments)]
fn validate_nemotron_experts(
    store: &SafetensorsWeightStore,
    args: &nemotron_h::ModelArgs,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    layer: usize,
    experts: usize,
    hidden: usize,
    intermediate: usize,
) {
    let official = format!("backbone.layers.{layer}.mixer.experts");
    let canonical = format!("model.layers.{layer}.moe.experts");
    validate_nemotron_experts_at(
        store,
        args,
        keys,
        allowed,
        issues,
        &format!("layer {layer}"),
        &official,
        &canonical,
        experts,
        hidden,
        intermediate,
    );
}

#[allow(clippy::too_many_arguments)]
fn validate_nemotron_experts_at(
    store: &SafetensorsWeightStore,
    args: &nemotron_h::ModelArgs,
    keys: &BTreeSet<String>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
    label: &str,
    official: &str,
    canonical: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
) {
    let packed_names = [
        format!("{official}.up_proj"),
        format!("{canonical}.up_proj"),
        format!("{official}.down_proj"),
        format!("{canonical}.down_proj"),
    ];
    let split_names = (0..experts)
        .flat_map(|expert| {
            [
                format!("{official}.{expert}.up_proj.weight"),
                format!("{canonical}.{expert}.up_proj.weight"),
                format!("{official}.{expert}.down_proj.weight"),
                format!("{canonical}.{expert}.down_proj.weight"),
            ]
        })
        .collect::<Vec<_>>();
    allowed.extend(packed_names.iter().cloned());
    allowed.extend(split_names.iter().cloned());
    let has_packed = packed_names.iter().any(|name| keys.contains(name));
    let has_split = split_names.iter().any(|name| keys.contains(name));
    if has_packed && has_split {
        let name = split_names
            .iter()
            .find(|name| keys.contains(*name))
            .cloned();
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!("Nemotron-H {label} mixes packed and split routed expert tensors"),
            tensor_name: name,
            tensor_type_code: None,
            metadata_key: None,
        });
        return;
    }
    if has_split && args.quantization.is_some() {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "checkpoint-native quantized Nemotron-H {label} requires packed routed expert banks"
            ),
            tensor_name: split_names
                .iter()
                .find(|name| keys.contains(*name))
                .cloned(),
            tensor_type_code: None,
            metadata_key: Some("quantization".into()),
        });
    }
    if has_packed {
        validate_nemotron_alias(
            store,
            args,
            keys,
            allowed,
            issues,
            format!("{official}.up_proj"),
            format!("{canonical}.up_proj"),
            vec![experts, intermediate, hidden],
        );
        validate_nemotron_alias(
            store,
            args,
            keys,
            allowed,
            issues,
            format!("{official}.down_proj"),
            format!("{canonical}.down_proj"),
            vec![experts, hidden, intermediate],
        );
    } else {
        for expert in 0..experts {
            validate_nemotron_alias(
                store,
                args,
                keys,
                allowed,
                issues,
                format!("{official}.{expert}.up_proj.weight"),
                format!("{canonical}.{expert}.up_proj.weight"),
                vec![intermediate, hidden],
            );
            validate_nemotron_alias(
                store,
                args,
                keys,
                allowed,
                issues,
                format!("{official}.{expert}.down_proj.weight"),
                format!("{canonical}.{expert}.down_proj.weight"),
                vec![hidden, intermediate],
            );
        }
    }
}

fn deepseek_v4_common_expected(args: &deepseek_v4::ModelArgs) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let hc_mult = args.hc_mult as usize;
    let hc_hidden = (args.hc_mult * args.hidden_size) as usize;
    let dense = |safetensors_name: String,
                 gguf_name: String,
                 shape: Vec<usize>,
                 operation: TensorOperation| ExpectedTensor {
        safetensors_name,
        gguf_name,
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation,
    };
    let mut tensors = vec![
        expected("embed.weight", "token_embd.weight", [vocab, hidden]),
        expected_vector("norm.weight", "output_norm.weight", hidden),
        expected("head.weight", "output.weight", [vocab, hidden]),
        expected("hc_head_fn", "output_hc_fn.weight", [hc_mult, hc_hidden]),
        expected_vector("hc_head_base", "output_hc_base.weight", hc_mult),
        expected_vector("hc_head_scale", "output_hc_scale.weight", 1),
    ];
    let mix = ((2 + args.hc_mult) * args.hc_mult) as usize;
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("layers.{layer}");
        let gguf = format!("blk.{layer}");
        let ratio = args.compress_ratios[layer];
        tensors.extend([
            expected(
                format!("{root}.attn.wq_a.weight"),
                format!("{gguf}.attn_q_a.weight"),
                [args.q_lora_rank as usize, hidden],
            ),
            expected_vector(
                format!("{root}.attn.q_norm.weight"),
                format!("{gguf}.attn_q_a_norm.weight"),
                args.q_lora_rank as usize,
            ),
            expected(
                format!("{root}.attn.wq_b.weight"),
                format!("{gguf}.attn_q_b.weight"),
                [
                    (args.num_attention_heads * args.head_dim) as usize,
                    args.q_lora_rank as usize,
                ],
            ),
            expected(
                format!("{root}.attn.wkv.weight"),
                format!("{gguf}.attn_kv.weight"),
                [args.head_dim as usize, hidden],
            ),
            expected_vector(
                format!("{root}.attn.kv_norm.weight"),
                format!("{gguf}.attn_kv_a_norm.weight"),
                args.head_dim as usize,
            ),
            expected(
                format!("{root}.attn.wo_a.weight"),
                format!("{gguf}.attn_output_a.weight"),
                [
                    (args.o_groups * args.o_lora_rank) as usize,
                    (args.num_attention_heads * args.head_dim / args.o_groups) as usize,
                ],
            ),
            expected(
                format!("{root}.attn.wo_b.weight"),
                format!("{gguf}.attn_output_b.weight"),
                [hidden, (args.o_groups * args.o_lora_rank) as usize],
            ),
            expected_vector(
                format!("{root}.attn.attn_sink"),
                format!("{gguf}.attn_sinks.weight"),
                args.num_attention_heads as usize,
            ),
            expected_vector(
                format!("{root}.attn_norm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{root}.ffn_norm.weight"),
                format!("{gguf}.ffn_norm.weight"),
                hidden,
            ),
            expected(
                format!("{root}.hc_attn_fn"),
                format!("{gguf}.hc_attn_fn.weight"),
                [mix, hc_hidden],
            ),
            expected_vector(
                format!("{root}.hc_attn_base"),
                format!("{gguf}.hc_attn_base.weight"),
                mix,
            ),
            expected_vector(
                format!("{root}.hc_attn_scale"),
                format!("{gguf}.hc_attn_scale.weight"),
                3,
            ),
            expected(
                format!("{root}.hc_ffn_fn"),
                format!("{gguf}.hc_ffn_fn.weight"),
                [mix, hc_hidden],
            ),
            expected_vector(
                format!("{root}.hc_ffn_base"),
                format!("{gguf}.hc_ffn_base.weight"),
                mix,
            ),
            expected_vector(
                format!("{root}.hc_ffn_scale"),
                format!("{gguf}.hc_ffn_scale.weight"),
                3,
            ),
            expected(
                format!("{root}.ffn.gate.weight"),
                format!("{gguf}.ffn_gate_inp.weight"),
                [args.n_routed_experts as usize, hidden],
            ),
        ]);
        if layer < args.num_hash_layers as usize {
            tensors.push(dense(
                format!("{root}.ffn.gate.tid2eid"),
                format!("{gguf}.ffn_gate_tid2eid.weight"),
                vec![vocab, args.num_experts_per_tok as usize],
                TensorOperation::I32,
            ));
        } else {
            tensors.push(expected_vector(
                format!("{root}.ffn.gate.bias"),
                format!("{gguf}.exp_probs_b.bias"),
                args.n_routed_experts as usize,
            ));
        }
        let shared = (args.moe_intermediate_size * args.n_shared_experts) as usize;
        tensors.extend([
            expected(
                format!("{root}.ffn.shared_experts.w1.weight"),
                format!("{gguf}.ffn_gate_shexp.weight"),
                [shared, hidden],
            ),
            expected(
                format!("{root}.ffn.shared_experts.w2.weight"),
                format!("{gguf}.ffn_down_shexp.weight"),
                [hidden, shared],
            ),
            expected(
                format!("{root}.ffn.shared_experts.w3.weight"),
                format!("{gguf}.ffn_up_shexp.weight"),
                [shared, hidden],
            ),
        ]);
        if ratio != 0 {
            let output = args.head_dim * if ratio == 4 { 2 } else { 1 };
            tensors.extend([
                expected(
                    format!("{root}.attn.compressor.wkv.weight"),
                    format!("{gguf}.attn_compressor_kv.weight"),
                    [output as usize, hidden],
                ),
                expected(
                    format!("{root}.attn.compressor.wgate.weight"),
                    format!("{gguf}.attn_compressor_gate.weight"),
                    [output as usize, hidden],
                ),
                dense(
                    format!("{root}.attn.compressor.ape"),
                    format!("{gguf}.attn_compressor_ape.weight"),
                    vec![ratio as usize, output as usize],
                    TensorOperation::Dense,
                ),
                expected_vector(
                    format!("{root}.attn.compressor.norm.weight"),
                    format!("{gguf}.attn_compressor_norm.weight"),
                    args.head_dim as usize,
                ),
            ]);
        }
        if ratio == 4 {
            let index_output = (2 * args.index_head_dim) as usize;
            tensors.extend([
                expected(
                    format!("{root}.attn.indexer.wq_b.weight"),
                    format!("{gguf}.indexer.attn_q_b.weight"),
                    [
                        (args.index_n_heads * args.index_head_dim) as usize,
                        args.q_lora_rank as usize,
                    ],
                ),
                expected(
                    format!("{root}.attn.indexer.weights_proj.weight"),
                    format!("{gguf}.indexer.proj.weight"),
                    [args.index_n_heads as usize, hidden],
                ),
                expected(
                    format!("{root}.attn.indexer.compressor.wkv.weight"),
                    format!("{gguf}.indexer_compressor_kv.weight"),
                    [index_output, hidden],
                ),
                expected(
                    format!("{root}.attn.indexer.compressor.wgate.weight"),
                    format!("{gguf}.indexer_compressor_gate.weight"),
                    [index_output, hidden],
                ),
                dense(
                    format!("{root}.attn.indexer.compressor.ape"),
                    format!("{gguf}.indexer_compressor_ape.weight"),
                    vec![ratio as usize, index_output],
                    TensorOperation::Dense,
                ),
                expected_vector(
                    format!("{root}.attn.indexer.compressor.norm.weight"),
                    format!("{gguf}.indexer_compressor_norm.weight"),
                    args.index_head_dim as usize,
                ),
            ]);
        }
    }
    tensors
}

fn validate_deepseek_v4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match deepseek_v4::ModelArgs::from_value(config.clone()) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let hidden = args.hidden_size as usize;
    let mut issues = Vec::new();
    for tensor in deepseek_v4_common_expected(&args) {
        if tensor.operation == TensorOperation::I32 {
            match store.metadata(&tensor.safetensors_name) {
                Ok(metadata) => {
                    if metadata.shape != tensor.safetensors_shape {
                        issues.push(shape_mismatch(
                            &tensor.safetensors_name,
                            &tensor.safetensors_shape,
                            &metadata.shape,
                        ));
                    }
                    if metadata.stored_dtype != StoredDtype::I32 {
                        issues.push(StructuralIssue {
                            kind: StructuralIssueKind::UnsupportedEncoding,
                            detail: format!(
                                "tensor {:?} must use I32, got {:?}",
                                tensor.safetensors_name, metadata.stored_dtype
                            ),
                            tensor_name: Some(tensor.safetensors_name),
                            tensor_type_code: None,
                            metadata_key: None,
                        });
                    }
                }
                Err(_) => issues.push(missing(&tensor.safetensors_name)),
            }
        } else {
            validate_safetensor(
                store,
                &tensor.safetensors_name,
                &tensor.safetensors_shape,
                false,
                &mut issues,
            );
        }
    }
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("layers.{layer}");
        for expert in 0..args.n_routed_experts as usize {
            for (projection, shape) in [
                ("w1", vec![args.moe_intermediate_size as usize, hidden]),
                ("w2", vec![hidden, args.moe_intermediate_size as usize]),
                ("w3", vec![args.moe_intermediate_size as usize, hidden]),
            ] {
                validate_safetensor(
                    store,
                    &format!("{root}.ffn.experts.{expert}.{projection}.weight"),
                    &shape,
                    false,
                    &mut issues,
                );
            }
        }
    }
    finish(issues)
}

fn validate_deepseek_v3_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let args = match deepseek_v3::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let affine = match args.affine_quantization() {
        Ok(affine) => affine,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let mtp_count = args.num_nextn_predict_layers as usize;
    let mut validation_args = args.clone();
    if mtp_count > 0 {
        let mut policies = args.layer_schedule.iter().copied().collect::<Vec<_>>();
        policies.extend(std::iter::repeat_n(
            deepseek_v3::LayerPolicy::SparseMoe,
            mtp_count,
        ));
        validation_args.layer_schedule = match crate::LayerSchedule::new(policies.len(), policies) {
            Ok(schedule) => schedule,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    }
    let mut expected_tensors = deepseek_v3_expected(&validation_args);
    for index in 0..mtp_count {
        let global = args.num_hidden_layers as usize + index;
        let prefix = format!("model.layers.{global}");
        expected_tensors.extend([
            expected_vector(
                format!("{prefix}.enorm.weight"),
                "",
                args.hidden_size as usize,
            ),
            expected_vector(
                format!("{prefix}.hnorm.weight"),
                "",
                args.hidden_size as usize,
            ),
            expected(
                format!("{prefix}.eh_proj.weight"),
                "",
                [args.hidden_size as usize, args.hidden_size as usize * 2],
            ),
            expected_vector(
                format!("{prefix}.shared_head.norm.weight"),
                "",
                args.hidden_size as usize,
            ),
            expected(
                format!("{prefix}.shared_head.head.weight"),
                "",
                [args.vocab_size as usize, args.hidden_size as usize],
            ),
        ]);
    }
    let native_fp8 = args.native_fp8_config().is_some();
    let deepseek_format = |name: &str| {
        if name.ends_with(".mlp.gate.weight") {
            SafetensorsMatrixFormat::Dense
        } else if native_fp8 && (name == "model.embed_tokens.weight" || name == "lm_head.weight") {
            // The native FP8 route leaves token embeddings and the output head
            // unquantized.  Affine checkpoints, however, pass both through the
            // ordinary MaybeQuantized embedding/linear constructors.
            SafetensorsMatrixFormat::Dense
        } else if native_fp8 {
            SafetensorsMatrixFormat::Fp8Block128
        } else if let Some(affine) = affine {
            SafetensorsMatrixFormat::Affine(affine)
        } else {
            SafetensorsMatrixFormat::Dense
        }
    };
    let mut allowed = BTreeSet::new();
    for tensor in &expected_tensors {
        allowed.insert(tensor.safetensors_name.clone());
        add_safetensors_format_companions(
            &mut allowed,
            &tensor.safetensors_name,
            deepseek_format(&tensor.safetensors_name),
        );
    }
    let mut issues = Vec::new();
    append_structural_issues(
        validate_safetensor_format_plan(store, expected_tensors, deepseek_format),
        &mut issues,
    );
    let allow_packed = !options.weight_residency.is_fully_resident();
    for (layer, policy) in validation_args.layer_schedule.iter().enumerate() {
        if *policy != deepseek_v3::LayerPolicy::SparseMoe {
            continue;
        }
        validate_deepseek_experts(
            store,
            &format!("model.layers.{layer}.mlp.experts"),
            args.n_routed_experts as usize,
            args.hidden_size as usize,
            args.moe_intermediate_size as usize,
            allow_packed,
            if native_fp8 {
                SafetensorsMatrixFormat::Fp8Block128
            } else if let Some(affine) = affine {
                SafetensorsMatrixFormat::Affine(affine)
            } else {
                SafetensorsMatrixFormat::Dense
            },
            &mut allowed,
            &mut issues,
        );
    }

    for key in store.keys() {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(&key, "DeepSeek-V3 SafeTensors"));
        }
    }
    finish(issues)
}

#[allow(clippy::too_many_arguments)]
fn validate_deepseek_experts(
    store: &SafetensorsWeightStore,
    prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    allow_packed: bool,
    format: SafetensorsMatrixFormat,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
) {
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    for (projection, shape, packed_shape) in [
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
        let packed = format!("{prefix}.{projection}");
        let split = (0..experts)
            .map(|expert| format!("{prefix}.{expert}.{projection}.weight"))
            .collect::<Vec<_>>();
        let present_split = split.iter().filter(|name| keys.contains(*name)).count();
        if keys.contains(&packed) {
            allowed.insert(packed.clone());
            add_safetensors_format_companions(allowed, &packed, format);
            if !allow_packed {
                issues.push(StructuralIssue {
                    kind: StructuralIssueKind::ConflictingLayout,
                    detail: format!(
                        "the requested resident DeepSeek-V3 loader requires per-expert {projection} tensors, not packed bank {packed:?}"
                    ),
                    tensor_name: Some(packed.clone()),
                    tensor_type_code: None,
                    metadata_key: None,
                });
            }
            if present_split > 0 {
                issues.push(StructuralIssue {
                    kind: StructuralIssueKind::ConflictingLayout,
                    detail: format!(
                        "DeepSeek-V3 expert catalog mixes packed bank {packed:?} with split {projection} tensors"
                    ),
                    tensor_name: split.iter().find(|name| keys.contains(*name)).cloned(),
                    tensor_type_code: None,
                    metadata_key: None,
                });
            }
            validate_safetensor_format(store, &packed, &packed_shape, format, issues);
            for name in split.into_iter().filter(|name| keys.contains(name)) {
                allowed.insert(name);
            }
        } else {
            for name in split {
                allowed.insert(name.clone());
                add_safetensors_format_companions(allowed, &name, format);
                validate_safetensor_format(store, &name, &shape, format, issues);
            }
        }
    }
}

fn validate_lfm2_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let args = match lfm2::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }
    let mut expected = match lfm2_expected(&args) {
        Ok(expected) => expected,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let mlx_conv_shape = [args.hidden_size as usize, args.conv_l_cache as usize, 1];
    for tensor in &mut expected {
        if tensor.safetensors_name.ends_with(".conv.conv.weight")
            && store
                .metadata(&tensor.safetensors_name)
                .is_ok_and(|metadata| metadata.shape == mlx_conv_shape)
        {
            // MLX-LM may serialize its native Conv1d layout. The LFM2 loaders
            // normalize this to SafeMLX's internal checkpoint layout.
            tensor.safetensors_shape = mlx_conv_shape.to_vec();
        }
    }
    if !args.has_sparse_moe_layers() {
        return validate_safetensor_plan_with(store, expected, |name| {
            args.weight_quantization_for(name)
        });
    }
    expected.retain(|tensor| !tensor.safetensors_name.contains(".feed_forward.experts."));
    let mut issues = Vec::new();
    append_structural_issues(
        validate_safetensor_plan_with(store, expected, |name| {
            if name.ends_with(".feed_forward.gate.weight") {
                None
            } else {
                args.weight_quantization_for(name)
            }
        }),
        &mut issues,
    );
    let allow_derived_packed = !options.weight_residency.is_fully_resident();
    for (layer, _) in args
        .layer_schedule
        .iter()
        .enumerate()
        .filter(|(_, policy)| policy.feed_forward == lfm2::FeedForwardPolicy::SparseMoe)
    {
        validate_split_or_packed_swiglu_experts(
            store,
            &format!("model.layers.{layer}.feed_forward.experts"),
            args.num_experts as usize,
            args.hidden_size as usize,
            args.moe_intermediate_size as usize,
            true,
            allow_derived_packed,
            args.weight_quantization_for(&format!(
                "model.layers.{layer}.feed_forward.experts.gate_up_proj"
            )),
            args.weight_quantization_for(&format!(
                "model.layers.{layer}.feed_forward.experts.down_proj"
            )),
            &mut issues,
        );
    }
    finish(issues)
}

fn validate_gpt_oss_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match gpt_oss::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }
    let common = gpt_oss_common_expected(&args);
    let mut allowed = BTreeSet::new();
    for tensor in &common {
        allowed.insert(tensor.safetensors_name.clone());
        if tensor.operation == TensorOperation::Matrix {
            if let Some(quantization) = args.quantization {
                add_safetensors_format_companions(
                    &mut allowed,
                    &tensor.safetensors_name,
                    SafetensorsMatrixFormat::Affine(quantization),
                );
            }
        }
    }
    let mut issues = match validate_safetensor_plan(store, common, args.quantization) {
        StructuralValidation::Exact => Vec::new(),
        StructuralValidation::Invalid(issues) => issues,
        StructuralValidation::Unverified(_) => {
            unreachable!("pure plan is always exact or invalid")
        }
    };
    let experts = args.num_local_experts as usize;
    let hidden = args.hidden_size as usize;
    let intermediate = args.intermediate_size as usize;
    for layer in 0..args.num_hidden_layers as usize {
        let prefix = format!("model.layers.{layer}.mlp.experts");
        for suffix in [
            "gate_up_proj_blocks",
            "gate_up_proj_scales",
            "gate_up_proj_bias",
            "down_proj_blocks",
            "down_proj_scales",
            "down_proj_bias",
        ] {
            allowed.insert(format!("{prefix}.{suffix}"));
        }
        validate_native_mxfp4_tensor(
            store,
            &format!("{prefix}.gate_up_proj_blocks"),
            &[experts, 2 * intermediate, hidden / 32, 16],
            false,
            &mut issues,
        );
        validate_native_mxfp4_tensor(
            store,
            &format!("{prefix}.gate_up_proj_scales"),
            &[experts, 2 * intermediate, hidden / 32],
            true,
            &mut issues,
        );
        validate_native_mxfp4_bias(
            store,
            &format!("{prefix}.gate_up_proj_bias"),
            &[experts, 2 * intermediate],
            &mut issues,
        );
        validate_native_mxfp4_tensor(
            store,
            &format!("{prefix}.down_proj_blocks"),
            &[experts, hidden, intermediate / 32, 16],
            false,
            &mut issues,
        );
        validate_native_mxfp4_tensor(
            store,
            &format!("{prefix}.down_proj_scales"),
            &[experts, hidden, intermediate / 32],
            true,
            &mut issues,
        );
        validate_native_mxfp4_bias(
            store,
            &format!("{prefix}.down_proj_bias"),
            &[experts, hidden],
            &mut issues,
        );
    }
    for name in store.keys() {
        if !allowed.contains(&name) {
            issues.push(unexpected_layout(&name, "GPT-OSS SafeTensors"));
        }
    }
    finish(issues)
}

fn validate_kimi_linear_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match kimi_linear::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }

    let mut issues = Vec::new();
    let mut normalized = BTreeMap::<String, String>::new();
    for raw in store.keys() {
        if raw.starts_with("model.mtp.") {
            continue;
        }
        let canonical = raw
            .replace(".block_sparse_moe.", ".mlp.")
            .replace(".inner.weight", ".weight");
        if let Some(previous) = normalized.insert(canonical.clone(), raw.clone()) {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::ConflictingLayout,
                detail: format!(
                    "Kimi Linear checkpoint tensors {previous:?} and {raw:?} both map to {canonical:?}"
                ),
                tensor_name: Some(raw),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
    }

    let mut expected = kimi_linear_expected(&args);
    let mut allowed = BTreeSet::new();
    for tensor in &mut expected {
        if let Some(raw) = normalized.get(&tensor.safetensors_name) {
            tensor.safetensors_name = raw.clone();
        }
        allowed.insert(tensor.safetensors_name.clone());
        if tensor.operation == TensorOperation::Matrix {
            let canonical = tensor
                .safetensors_name
                .replace(".block_sparse_moe.", ".mlp.")
                .replace(".inner.weight", ".weight");
            if let Some(quantization) = args.weight_quantization_for(&canonical) {
                add_safetensors_format_companions(
                    &mut allowed,
                    &tensor.safetensors_name,
                    SafetensorsMatrixFormat::Affine(quantization),
                );
            }
        }
    }
    append_structural_issues(
        validate_safetensor_plan_with(store, expected, |raw| {
            args.weight_quantization_for(
                &raw.replace(".block_sparse_moe.", ".mlp.")
                    .replace(".inner.weight", ".weight"),
            )
        }),
        &mut issues,
    );

    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.attention == kimi_linear::AttentionKind::Kda {
            let prefix = format!("model.layers.{layer}.self_attn");
            let projection = (args.kda_config.num_heads * args.kda_config.head_dim) as usize;
            let kernel = args.kda_config.short_conv_kernel_size as usize;
            for name in ["q_conv1d.weight", "k_conv1d.weight", "v_conv1d.weight"] {
                let canonical = format!("{prefix}.{name}");
                let raw = normalized.get(&canonical).unwrap_or(&canonical);
                allowed.insert(raw.clone());
                validate_float_element_count(store, raw, projection * kernel, &mut issues);
            }
            let canonical = format!("{prefix}.A_log");
            let raw = normalized.get(&canonical).unwrap_or(&canonical);
            allowed.insert(raw.clone());
            validate_float_element_count(
                store,
                raw,
                args.kda_config.num_heads as usize,
                &mut issues,
            );
        }
        if policy.feed_forward != kimi_linear::FeedForwardPolicy::SparseMoe {
            continue;
        }
        let prefix = format!("model.layers.{layer}.mlp.experts");
        let gate_up = format!("{prefix}.gate_up_proj");
        let down = format!("{prefix}.down_proj");
        let has_packed = normalized.contains_key(&gate_up) || normalized.contains_key(&down);
        let mut expert_expected = Vec::new();
        if has_packed {
            let gate_up_raw = normalized.get(&gate_up).unwrap_or(&gate_up).clone();
            let down_raw = normalized.get(&down).unwrap_or(&down).clone();
            allowed.extend([gate_up_raw.clone(), down_raw.clone()]);
            for (raw, canonical) in [(&gate_up_raw, &gate_up), (&down_raw, &down)] {
                if let Some(quantization) = args.weight_quantization_for(canonical) {
                    allowed.insert(format!("{raw}.scales"));
                    if quantization.has_biases() {
                        allowed.insert(format!("{raw}.biases"));
                    }
                }
            }
            expert_expected.extend([
                expected_rank3(
                    gate_up_raw,
                    "",
                    [
                        args.num_experts as usize,
                        2 * args.moe_intermediate_size as usize,
                        args.hidden_size as usize,
                    ],
                ),
                expected_rank3(
                    down_raw,
                    "",
                    [
                        args.num_experts as usize,
                        args.hidden_size as usize,
                        args.moe_intermediate_size as usize,
                    ],
                ),
            ]);
        } else {
            if args.quantization.is_some() {
                issues.push(StructuralIssue {
                    kind: StructuralIssueKind::ConflictingLayout,
                    detail: format!(
                        "checkpoint-native quantized Kimi Linear layer {layer} requires packed expert banks"
                    ),
                    tensor_name: Some(format!("{prefix}.0.w1.weight")),
                    tensor_type_code: None,
                    metadata_key: Some("quantization".into()),
                });
            }
            for expert in 0..args.num_experts as usize {
                for (projection, shape) in [
                    (
                        "w1",
                        vec![
                            args.moe_intermediate_size as usize,
                            args.hidden_size as usize,
                        ],
                    ),
                    (
                        "w2",
                        vec![
                            args.hidden_size as usize,
                            args.moe_intermediate_size as usize,
                        ],
                    ),
                    (
                        "w3",
                        vec![
                            args.moe_intermediate_size as usize,
                            args.hidden_size as usize,
                        ],
                    ),
                ] {
                    let canonical = format!("{prefix}.{expert}.{projection}.weight");
                    let raw = normalized.get(&canonical).unwrap_or(&canonical).clone();
                    allowed.insert(raw.clone());
                    expert_expected.push(ExpectedTensor {
                        safetensors_name: raw,
                        gguf_name: String::new(),
                        safetensors_shape: shape.clone(),
                        gguf_shape: shape,
                        operation: TensorOperation::Matrix,
                    });
                }
            }
        }
        append_structural_issues(
            validate_safetensor_plan_with(store, expert_expected, |raw| {
                let canonical = raw.replace(".block_sparse_moe.", ".mlp.");
                if canonical.ends_with("gate_up_proj") || canonical.ends_with("down_proj") {
                    args.weight_quantization_for(&canonical)
                } else {
                    None
                }
            }),
            &mut issues,
        );
    }

    for raw in store.keys() {
        if !raw.starts_with("model.mtp.") && !allowed.contains(&raw) {
            issues.push(unexpected_layout(&raw, "Kimi Linear SafeTensors"));
        }
    }
    finish(issues)
}

fn append_structural_issues(validation: StructuralValidation, issues: &mut Vec<StructuralIssue>) {
    match validation {
        StructuralValidation::Exact => {}
        StructuralValidation::Invalid(found) => issues.extend(found),
        StructuralValidation::Unverified(_) => {
            unreachable!("pure tensor plan is always exact or invalid")
        }
    }
}

fn validate_float_element_count(
    store: &SafetensorsWeightStore,
    name: &str,
    elements: usize,
    issues: &mut Vec<StructuralIssue>,
) {
    let metadata = match store.metadata(name) {
        Ok(metadata) => metadata,
        Err(crate::runtime::checkpoint::store::WeightStoreError::UnknownTensor { .. }) => {
            issues.push(missing(name));
            return;
        }
        Err(error) => {
            issues.push(layout(name, error.to_string()));
            return;
        }
    };
    let actual = metadata
        .shape
        .iter()
        .try_fold(1usize, |total, dimension| total.checked_mul(*dimension));
    if actual != Some(elements) {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ShapeMismatch,
            detail: format!(
                "tensor {name:?} must contain {elements} elements for the loader reshape, got shape {:?}",
                metadata.shape
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
    if !is_float_dtype(&metadata.stored_dtype) {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::UnsupportedEncoding,
            detail: format!(
                "tensor {name:?} uses unsupported SafeTensors dtype {:?}",
                metadata.stored_dtype
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
}

fn validate_native_mxfp4_tensor(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    companion: bool,
    issues: &mut Vec<StructuralIssue>,
) {
    let metadata = match store.metadata(name) {
        Ok(metadata) => metadata,
        Err(crate::runtime::checkpoint::store::WeightStoreError::UnknownTensor { .. })
            if companion =>
        {
            issues.push(quantization_companion_issue(
                name,
                format!("native MXFP4 expert weight is missing required companion {name:?}"),
            ));
            return;
        }
        Err(crate::runtime::checkpoint::store::WeightStoreError::UnknownTensor { .. }) => {
            issues.push(missing(name));
            return;
        }
        Err(error) => {
            issues.push(layout(name, error.to_string()));
            return;
        }
    };
    if metadata.shape != shape || !matches!(metadata.stored_dtype, StoredDtype::U8) {
        let detail = format!(
            "native MXFP4 tensor {name:?} expected shape {shape:?} and U8 storage, got {:?} {:?}",
            metadata.shape, metadata.stored_dtype
        );
        if companion {
            issues.push(quantization_companion_issue(name, detail));
        } else {
            if metadata.shape != shape {
                issues.push(shape_mismatch(name, shape, &metadata.shape));
            }
            if !matches!(metadata.stored_dtype, StoredDtype::U8) {
                issues.push(StructuralIssue {
                    kind: StructuralIssueKind::UnsupportedEncoding,
                    detail,
                    tensor_name: Some(name.into()),
                    tensor_type_code: None,
                    metadata_key: Some("quantization_config.quant_method".into()),
                });
            }
        }
    }
}

fn validate_native_mxfp4_bias(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    issues: &mut Vec<StructuralIssue>,
) {
    match store.metadata(name) {
        Ok(metadata)
            if metadata.shape == shape && is_float_dtype(&metadata.stored_dtype) => {}
        Ok(metadata) => issues.push(quantization_companion_issue(
            name,
            format!(
                "native MXFP4 bias {name:?} expected shape {shape:?} and floating storage, got {:?} {:?}",
                metadata.shape, metadata.stored_dtype
            ),
        )),
        Err(_) => issues.push(quantization_companion_issue(
            name,
            format!("native MXFP4 expert weight is missing required bias {name:?}"),
        )),
    }
}

fn quantization_companion_issue(name: &str, detail: String) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::QuantizationCompanionMismatch,
        detail,
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: Some("quantization_config.quant_method".into()),
    }
}

fn validate_llama_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match llama::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }
    let expected = llama_expected(&args);
    let quantization = args.weight_quantization();
    let mut allowed = BTreeSet::new();
    for tensor in &expected {
        allowed.insert(tensor.safetensors_name.clone());
        if tensor.operation == TensorOperation::Matrix {
            if let Some(quantization) = quantization {
                add_safetensors_format_companions(
                    &mut allowed,
                    &tensor.safetensors_name,
                    SafetensorsMatrixFormat::Affine(quantization),
                );
            }
        }
    }
    let mut issues = match validate_safetensor_plan(store, expected, quantization) {
        StructuralValidation::Exact => Vec::new(),
        StructuralValidation::Invalid(issues) => issues,
        StructuralValidation::Unverified(_) => {
            unreachable!("pure plan is always exact or invalid")
        }
    };
    for key in store.keys() {
        if !allowed.contains(&key)
            && !key.starts_with("rope_freqs.")
            && !key.ends_with(".rotary_emb.inv_freq")
        {
            issues.push(unexpected_layout(&key, "Llama SafeTensors"));
        }
    }
    finish(issues)
}

fn validate_qwen3_next_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_qwen3_next_safetensors(
        config,
        store,
        options.weight_residency.expert_cache().is_some(),
    )
}

fn validate_qwen35_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_qwen35_safetensors(
        config,
        store,
        options.weight_residency.expert_cache().is_some(),
    )
}

fn validate_dense_qwen_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let args = match dense_qwen::config_from_hf_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > store.keys().len() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor checkpoint catalog",
            args.num_hidden_layers,
            store.keys().len()
        ));
    }
    let mut expected = dense_qwen_expected(&args);
    if !args.is_moe() {
        let mut allowed = BTreeSet::new();
        for tensor in &expected {
            allowed.insert(tensor.safetensors_name.clone());
            if tensor.operation == TensorOperation::Matrix {
                if let Some(quantization) = args.weight_quantization_for(&tensor.safetensors_name) {
                    add_safetensors_format_companions(
                        &mut allowed,
                        &tensor.safetensors_name,
                        SafetensorsMatrixFormat::Affine(quantization),
                    );
                }
            }
        }
        let mut issues = Vec::new();
        append_structural_issues(
            validate_safetensor_plan_with(store, expected, |name| {
                args.weight_quantization_for(name)
            }),
            &mut issues,
        );
        for name in store.keys() {
            if !allowed.contains(&name) {
                issues.push(unexpected_layout(&name, "dense-Qwen SafeTensors"));
            }
        }
        return finish(issues);
    }
    expected.retain(|tensor| !tensor.safetensors_name.contains(".mlp.experts."));
    let mut issues = Vec::new();
    append_structural_issues(
        validate_safetensor_plan_with(store, expected, |name| {
            if name.ends_with(".mlp.gate.weight") {
                None
            } else {
                args.weight_quantization_for(name)
            }
        }),
        &mut issues,
    );
    for layer in 0..args.num_hidden_layers as usize {
        validate_split_or_packed_swiglu_experts(
            store,
            &format!("model.layers.{layer}.mlp.experts"),
            args.num_experts as usize,
            args.hidden_size as usize,
            args.moe_intermediate_size as usize,
            true,
            true,
            args.weight_quantization_for(&format!("model.layers.{layer}.mlp.experts.gate_up_proj")),
            args.weight_quantization_for(&format!("model.layers.{layer}.mlp.experts.down_proj")),
            &mut issues,
        );
    }
    finish(issues)
}

fn validate_muse_glimmer_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let expected = match muse_glimmer_expected(config) {
        Ok(expected) => expected,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match muse_glimmer::config_from_hf_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let quantization = args.weight_quantization();
    let mut allowed = BTreeSet::new();
    for tensor in &expected {
        allowed.insert(tensor.safetensors_name.clone());
        if tensor.operation == TensorOperation::Matrix {
            if let Some(quantization) = quantization {
                add_safetensors_format_companions(
                    &mut allowed,
                    &tensor.safetensors_name,
                    SafetensorsMatrixFormat::Affine(quantization),
                );
            }
        }
    }
    let mut issues = Vec::new();
    append_structural_issues(
        validate_safetensor_plan(store, expected, quantization),
        &mut issues,
    );
    for key in store.keys() {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(&key, "Muse-Glimmer SafeTensors"));
        }
    }
    finish(issues)
}

fn validate_personaplex_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    let metadata = match personaplex::model_metadata_from_config_value(config) {
        Ok(metadata) => metadata,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let mut args = personaplex::model_args_7b_v1();
    args.quantization = metadata.quantization;
    if args.num_layers as usize > store.keys().len()
        || args.depformer_num_layers as usize > store.keys().len()
    {
        return invalid_geometry(format!(
            "configured PersonaPlex temporal/depth layer counts {}/{} exceed the entire {}-tensor checkpoint catalog",
            args.num_layers,
            args.depformer_num_layers,
            store.keys().len()
        ));
    }

    let quantization = args.quantization;
    let dim = args.dim as usize;
    let depth_dim = args.depformer_dim as usize;
    let temporal_hidden = moshi_mlp_hidden(dim, args.dim_feedforward.map(|value| value as usize));
    let depth_hidden = moshi_mlp_hidden(
        depth_dim,
        args.depformer_dim_feedforward.map(|value| value as usize),
    );
    let mut allowed = BTreeSet::new();
    let mut issues = Vec::new();

    for (name, shape) in [
        (
            "text_emb.weight".to_string(),
            vec![args.text_card as usize + 1, dim],
        ),
        (
            "text_linear.weight".to_string(),
            vec![args.text_card as usize, dim],
        ),
    ] {
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&name),
            name.trim_end_matches(".weight"),
            &shape,
            quantization,
            &mut allowed,
            &mut issues,
        );
    }
    validate_personaplex_norm(store, "out_norm.alpha", dim, &mut allowed, &mut issues);
    for codebook in 0..args.n_q as usize {
        let name = format!("emb.{codebook}.weight");
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&name),
            name.trim_end_matches(".weight"),
            &[args.card as usize + 1, dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
    }

    for layer in 0..args.num_layers as usize {
        let prefix = format!("transformer.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            validate_personaplex_norm(
                store,
                &format!("{prefix}.{norm}.alpha"),
                dim,
                &mut allowed,
                &mut issues,
            );
        }
        let in_proj = [
            format!("{prefix}.self_attn.in_proj_weight"),
            format!("{prefix}.self_attn.in_proj.weight"),
        ];
        validate_personaplex_matrix(
            store,
            &in_proj,
            &format!("{prefix}.self_attn.in_proj"),
            &[3 * dim, dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        for (name, shape) in [
            (
                format!("{prefix}.self_attn.out_proj.weight"),
                vec![dim, dim],
            ),
            (
                format!("{prefix}.gating.linear_in.weight"),
                vec![2 * temporal_hidden, dim],
            ),
            (
                format!("{prefix}.gating.linear_out.weight"),
                vec![dim, temporal_hidden],
            ),
        ] {
            validate_personaplex_matrix(
                store,
                std::slice::from_ref(&name),
                name.trim_end_matches(".weight"),
                &shape,
                quantization,
                &mut allowed,
                &mut issues,
            );
        }
    }

    for slice in 0..args.dep_q as usize {
        let embedding = if slice == 0 {
            "depformer_text_emb.weight".to_string()
        } else {
            format!("depformer_emb.{}.weight", slice - 1)
        };
        let input_vocab = if slice == 0 {
            args.text_card as usize + 1
        } else {
            args.card as usize + 1
        };
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&embedding),
            embedding.trim_end_matches(".weight"),
            &[input_vocab, depth_dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        for (name, shape) in [
            (format!("depformer_in.{slice}.weight"), vec![depth_dim, dim]),
            (
                format!("linears.{slice}.weight"),
                vec![args.card as usize, depth_dim],
            ),
        ] {
            validate_personaplex_matrix(
                store,
                std::slice::from_ref(&name),
                name.trim_end_matches(".weight"),
                &shape,
                quantization,
                &mut allowed,
                &mut issues,
            );
        }
    }

    let depth_slices = args.dep_q as usize;
    for layer in 0..args.depformer_num_layers as usize {
        let prefix = format!("depformer.layers.{layer}");
        for norm in ["norm1", "norm2"] {
            validate_personaplex_norm(
                store,
                &format!("{prefix}.{norm}.alpha"),
                depth_dim,
                &mut allowed,
                &mut issues,
            );
        }
        let in_proj = [
            format!("{prefix}.self_attn.in_proj_weight"),
            format!("{prefix}.self_attn.in_proj.weight"),
        ];
        validate_personaplex_matrix(
            store,
            &in_proj,
            &format!("{prefix}.self_attn.in_proj"),
            &[depth_slices * 3 * depth_dim, depth_dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        let out_proj = format!("{prefix}.self_attn.out_proj.weight");
        validate_personaplex_matrix(
            store,
            std::slice::from_ref(&out_proj),
            out_proj.trim_end_matches(".weight"),
            &[depth_slices * depth_dim, depth_dim],
            quantization,
            &mut allowed,
            &mut issues,
        );
        for slice in 0..depth_slices {
            for (name, shape) in [
                (
                    format!("{prefix}.gating.{slice}.linear_in.weight"),
                    vec![2 * depth_hidden, depth_dim],
                ),
                (
                    format!("{prefix}.gating.{slice}.linear_out.weight"),
                    vec![depth_dim, depth_hidden],
                ),
            ] {
                validate_personaplex_matrix(
                    store,
                    std::slice::from_ref(&name),
                    name.trim_end_matches(".weight"),
                    &shape,
                    quantization,
                    &mut allowed,
                    &mut issues,
                );
            }
        }
    }

    for key in store.keys() {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(
                &key,
                "PersonaPlex released PyTorch SafeTensors",
            ));
        }
    }
    finish(issues)
}

fn moshi_mlp_hidden(dim: usize, feed_forward: Option<usize>) -> usize {
    let feed_forward = feed_forward.unwrap_or(4 * dim);
    if feed_forward == 4 * dim {
        11 * dim / 4
    } else {
        2 * feed_forward / 3
    }
}

#[allow(clippy::too_many_arguments)]
fn validate_personaplex_matrix(
    store: &SafetensorsWeightStore,
    aliases: &[String],
    companion_prefix: &str,
    shape: &[usize],
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
) {
    allowed.extend(aliases.iter().cloned());
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let present = aliases
        .iter()
        .filter(|name| keys.contains(*name))
        .collect::<Vec<_>>();
    if present.len() > 1 {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "PersonaPlex checkpoint contains multiple aliases for {:?}: {present:?}",
                aliases[0]
            ),
            tensor_name: Some(present[1].clone()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
    let Some(name) = present.first().map(|name| name.as_str()) else {
        issues.push(missing(&aliases[0]));
        return;
    };
    let Some(quantization) = quantization else {
        validate_safetensor(store, name, shape, false, issues);
        return;
    };

    let input = *shape.last().expect("PersonaPlex matrix shape");
    let group_size = quantization.group_size() as usize;
    let bits = quantization.bits() as usize;
    if !input.is_multiple_of(group_size)
        || !input.is_multiple_of(32)
        || !input.saturating_mul(bits).is_multiple_of(32)
    {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::QuantizationCompanionMismatch,
            detail: format!(
                "quantized PersonaPlex tensor {name:?} input dimension {input} is incompatible with group size {group_size} and {bits}-bit packing"
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: Some("quantization".into()),
        });
        return;
    }
    let mut packed = shape.to_vec();
    *packed.last_mut().expect("PersonaPlex matrix shape") = input * bits / 32;
    validate_safetensor(store, name, &packed, true, issues);
    let mut companions = shape.to_vec();
    *companions.last_mut().expect("PersonaPlex matrix shape") = input / group_size;
    let scales = format!("{companion_prefix}.scales");
    allowed.insert(scales.clone());
    validate_quantization_companion(store, &scales, &companions, issues);
    if quantization.has_biases() {
        let biases = format!("{companion_prefix}.biases");
        allowed.insert(biases.clone());
        validate_quantization_companion(store, &biases, &companions, issues);
    }
}

fn validate_personaplex_norm(
    store: &SafetensorsWeightStore,
    name: &str,
    elements: usize,
    allowed: &mut BTreeSet<String>,
    issues: &mut Vec<StructuralIssue>,
) {
    allowed.insert(name.into());
    let metadata = match store.metadata(name) {
        Ok(metadata) => metadata,
        Err(crate::runtime::checkpoint::store::WeightStoreError::UnknownTensor { .. }) => {
            issues.push(missing(name));
            return;
        }
        Err(error) => {
            issues.push(layout(name, error.to_string()));
            return;
        }
    };
    if metadata.shape.iter().product::<usize>() != elements {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ShapeMismatch,
            detail: format!(
                "PersonaPlex norm tensor {name:?} must contain {elements} elements for loader reshape, got shape {:?}",
                metadata.shape
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
    if !is_float_dtype(&metadata.stored_dtype) {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::UnsupportedEncoding,
            detail: format!(
                "tensor {name:?} uses unsupported SafeTensors dtype {:?}",
                metadata.stored_dtype
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
}

fn validate_qwen3_vl_safetensors(
    kind: ModelKind,
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let args = match qwen3_vl::model_args_from_config_value(config) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let is_moe = args.text_config.is_moe();
    if is_moe != (kind == ModelKind::Qwen3VlMoe) {
        return invalid_geometry(format!(
            "Qwen3-VL dispatch selected {kind:?}, but the nested text configuration is {}",
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

    let quantization = args.text_config.weight_quantization();
    let (mut text, vision) = qwen3_vl_safetensors_expected(&args);
    let mut allowed = BTreeSet::new();
    for tensor in text.iter().chain(&vision) {
        allowed.insert(tensor.safetensors_name.clone());
        if tensor.operation == TensorOperation::Matrix {
            if let Some(quantization) = quantization {
                add_safetensors_format_companions(
                    &mut allowed,
                    &tensor.safetensors_name,
                    SafetensorsMatrixFormat::Affine(quantization),
                );
            }
        }
    }

    let mut issues = Vec::new();
    if is_moe {
        text.retain(|tensor| !tensor.safetensors_name.contains(".mlp.experts."));
    }
    append_structural_issues(
        validate_safetensor_plan_with(store, text, |name| {
            if name.ends_with(".mlp.gate.weight") {
                None
            } else {
                args.text_config.weight_quantization_for(name)
            }
        }),
        &mut issues,
    );
    append_structural_issues(validate_safetensor_plan(store, vision, None), &mut issues);

    if is_moe {
        let experts = args.text_config.num_experts as usize;
        let hidden = args.text_config.hidden_size as usize;
        let intermediate = args.text_config.moe_intermediate_size as usize;
        let allow_split = !options.weight_residency.is_fully_resident();
        for layer in 0..args.text_config.num_hidden_layers as usize {
            let prefix = format!("model.language_model.layers.{layer}.mlp.experts");
            validate_split_or_packed_swiglu_experts(
                store,
                &prefix,
                experts,
                hidden,
                intermediate,
                allow_split,
                allow_split,
                args.text_config
                    .weight_quantization_for(&format!("{prefix}.gate_up_proj")),
                args.text_config
                    .weight_quantization_for(&format!("{prefix}.down_proj")),
                &mut issues,
            );
            allowed.extend([
                format!("{prefix}.gate_up_proj"),
                format!("{prefix}.gate_proj"),
                format!("{prefix}.up_proj"),
                format!("{prefix}.down_proj"),
            ]);
            for bank in ["gate_up_proj", "gate_proj", "up_proj", "down_proj"] {
                let name = format!("{prefix}.{bank}");
                if let Some(quantization) = args.text_config.weight_quantization_for(&name) {
                    allowed.insert(format!("{name}.scales"));
                    if quantization.has_biases() {
                        allowed.insert(format!("{name}.biases"));
                    }
                }
            }
            for expert in 0..experts {
                for projection in ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"] {
                    allowed.insert(format!("{prefix}.{expert}.{projection}.weight"));
                }
            }
        }
    }

    for key in store.keys() {
        if !allowed.contains(&key) {
            issues.push(unexpected_layout(&key, "Qwen3-VL SafeTensors"));
        }
    }
    finish(issues)
}

fn qwen3_vl_safetensors_expected(
    args: &qwen3_vl::ModelArgs,
) -> (Vec<ExpectedTensor>, Vec<ExpectedTensor>) {
    let mut text = dense_qwen_expected(&args.text_config);
    for tensor in &mut text {
        if let Some(rest) = tensor.safetensors_name.strip_prefix("model.") {
            tensor.safetensors_name = format!("model.language_model.{rest}");
        }
    }

    let vision = qwen_vision_safetensors_expected(
        &args.vision_config,
        args.text_config.hidden_size as usize,
        "model.visual",
    );
    (text, vision)
}

fn qwen_vision_safetensors_expected(
    config: &qwen3_vl::VisionConfig,
    text_hidden: usize,
    root: &str,
) -> Vec<ExpectedTensor> {
    let hidden = config.hidden_size as usize;
    let intermediate = config.intermediate_size as usize;
    let channels = config.in_channels as usize;
    let temporal = config.temporal_patch_size as usize;
    let patch = config.patch_size as usize;
    let merger_hidden = hidden * (config.spatial_merge_size as usize).pow(2);
    let dense = |name: String, shape: Vec<usize>| ExpectedTensor {
        safetensors_name: name,
        gguf_name: String::new(),
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Dense,
    };
    let mut vision = vec![
        dense(
            format!("{root}.pos_embed.weight"),
            vec![config.num_position_embeddings as usize, hidden],
        ),
        dense(
            format!("{root}.patch_embed.proj.weight"),
            vec![hidden, channels, temporal, patch, patch],
        ),
        dense(format!("{root}.patch_embed.proj.bias"), vec![hidden]),
    ];
    for layer in 0..config.layer_count() {
        let prefix = format!("{root}.blocks.{layer}");
        vision.extend([
            dense(format!("{prefix}.norm1.weight"), vec![hidden]),
            dense(format!("{prefix}.norm1.bias"), vec![hidden]),
            dense(
                format!("{prefix}.attn.qkv.weight"),
                vec![3 * hidden, hidden],
            ),
            dense(format!("{prefix}.attn.qkv.bias"), vec![3 * hidden]),
            dense(format!("{prefix}.attn.proj.weight"), vec![hidden, hidden]),
            dense(format!("{prefix}.attn.proj.bias"), vec![hidden]),
            dense(format!("{prefix}.norm2.weight"), vec![hidden]),
            dense(format!("{prefix}.norm2.bias"), vec![hidden]),
            dense(
                format!("{prefix}.mlp.linear_fc1.weight"),
                vec![intermediate, hidden],
            ),
            dense(format!("{prefix}.mlp.linear_fc1.bias"), vec![intermediate]),
            dense(
                format!("{prefix}.mlp.linear_fc2.weight"),
                vec![hidden, intermediate],
            ),
            dense(format!("{prefix}.mlp.linear_fc2.bias"), vec![hidden]),
        ]);
    }
    vision.extend([
        dense(format!("{root}.merger.norm.weight"), vec![hidden]),
        dense(format!("{root}.merger.norm.bias"), vec![hidden]),
        dense(
            format!("{root}.merger.linear_fc1.weight"),
            vec![merger_hidden, merger_hidden],
        ),
        dense(
            format!("{root}.merger.linear_fc1.bias"),
            vec![merger_hidden],
        ),
        dense(
            format!("{root}.merger.linear_fc2.weight"),
            vec![text_hidden, merger_hidden],
        ),
        dense(format!("{root}.merger.linear_fc2.bias"), vec![text_hidden]),
    ]);
    for index in 0..config.deepstack_layer_count() {
        let prefix = format!("{root}.deepstack_merger_list.{index}");
        vision.extend([
            dense(format!("{prefix}.norm.weight"), vec![merger_hidden]),
            dense(format!("{prefix}.norm.bias"), vec![merger_hidden]),
            dense(
                format!("{prefix}.linear_fc1.weight"),
                vec![merger_hidden, merger_hidden],
            ),
            dense(format!("{prefix}.linear_fc1.bias"), vec![merger_hidden]),
            dense(
                format!("{prefix}.linear_fc2.weight"),
                vec![text_hidden, merger_hidden],
            ),
            dense(format!("{prefix}.linear_fc2.bias"), vec![text_hidden]),
        ]);
    }
    vision
}

#[allow(clippy::too_many_arguments)]
fn validate_split_or_packed_swiglu_experts(
    store: &SafetensorsWeightStore,
    prefix: &str,
    experts: usize,
    hidden: usize,
    intermediate: usize,
    allow_per_expert_split: bool,
    allow_separate_packed: bool,
    gate_up_quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    down_quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
    issues: &mut Vec<StructuralIssue>,
) {
    let keys = store.keys().into_iter().collect::<BTreeSet<_>>();
    let gate_up = format!("{prefix}.gate_up_proj");
    let gate = format!("{prefix}.gate_proj");
    let up = format!("{prefix}.up_proj");
    let down = format!("{prefix}.down_proj");
    let packed_present = keys.contains(&gate_up)
        || (keys.contains(&down) && !keys.contains(&gate) && !keys.contains(&up));
    if packed_present {
        let mut packed_allowed = BTreeSet::from([gate_up.clone(), down.clone()]);
        for (name, quantization) in [(&gate_up, gate_up_quantization), (&down, down_quantization)] {
            if let Some(quantization) = quantization {
                packed_allowed.insert(format!("{name}.scales"));
                if quantization.has_biases() {
                    packed_allowed.insert(format!("{name}.biases"));
                }
            }
        }
        for key in keys
            .iter()
            .filter(|key| key.starts_with(&format!("{prefix}.")) && !packed_allowed.contains(*key))
        {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::ConflictingLayout,
                detail: format!("expert catalog {prefix:?} mixes packed tensor names with {key:?}"),
                tensor_name: Some(key.clone()),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
        append_structural_issues(
            validate_safetensor_plan_with(
                store,
                vec![
                    expected_rank3(gate_up, "", [experts, 2 * intermediate, hidden]),
                    expected_rank3(down, "", [experts, hidden, intermediate]),
                ],
                |name| {
                    if name.ends_with("gate_up_proj") {
                        gate_up_quantization
                    } else {
                        down_quantization
                    }
                },
            ),
            issues,
        );
        return;
    }

    let separate_present = keys.contains(&gate) || keys.contains(&up) || keys.contains(&down);
    if separate_present {
        if !allow_separate_packed {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::ConflictingLayout,
                detail: format!(
                    "the requested resident loader requires {gate_up:?}; separate packed gate/up tensors are only supported by bounded loading"
                ),
                tensor_name: Some(gate),
                tensor_type_code: None,
                metadata_key: None,
            });
            issues.push(missing(&gate_up));
            return;
        }
        let mut separate_allowed = BTreeSet::from([gate.clone(), up.clone(), down.clone()]);
        for (name, quantization) in [
            (&gate, gate_up_quantization),
            (&up, gate_up_quantization),
            (&down, down_quantization),
        ] {
            if let Some(quantization) = quantization {
                separate_allowed.insert(format!("{name}.scales"));
                if quantization.has_biases() {
                    separate_allowed.insert(format!("{name}.biases"));
                }
            }
        }
        for key in keys.iter().filter(|key| {
            key.starts_with(&format!("{prefix}.")) && !separate_allowed.contains(*key)
        }) {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::ConflictingLayout,
                detail: format!(
                    "expert catalog {prefix:?} mixes separate packed banks with {key:?}"
                ),
                tensor_name: Some(key.clone()),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
        append_structural_issues(
            validate_safetensor_plan_with(
                store,
                vec![
                    expected_rank3(gate, "", [experts, intermediate, hidden]),
                    expected_rank3(up, "", [experts, intermediate, hidden]),
                    expected_rank3(down, "", [experts, hidden, intermediate]),
                ],
                |name| {
                    if name.ends_with("down_proj") {
                        down_quantization
                    } else {
                        gate_up_quantization
                    }
                },
            ),
            issues,
        );
        return;
    }

    if !allow_per_expert_split || gate_up_quantization.is_some() || down_quantization.is_some() {
        let split = keys
            .iter()
            .find(|key| key.starts_with(&format!("{prefix}.")))
            .cloned();
        if let Some(split) = split {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::ConflictingLayout,
                detail: format!(
                    "the selected loader cannot derive the required packed expert representation from {split:?}"
                ),
                tensor_name: Some(split),
                tensor_type_code: None,
                metadata_key: None,
            });
        }
        issues.extend([missing(&gate_up), missing(&down)]);
        return;
    }

    let mut allowed = BTreeSet::new();
    for expert in 0..experts {
        for (aliases, shape) in [
            (["w1", "gate_proj"], vec![intermediate, hidden]),
            (["w2", "down_proj"], vec![hidden, intermediate]),
            (["w3", "up_proj"], vec![intermediate, hidden]),
        ] {
            let candidates =
                aliases.map(|projection| format!("{prefix}.{expert}.{projection}.weight"));
            allowed.extend(candidates.iter().cloned());
            let present = candidates
                .iter()
                .filter(|candidate| keys.contains(*candidate))
                .collect::<Vec<_>>();
            if present.len() > 1 {
                issues.push(StructuralIssue {
                    kind: StructuralIssueKind::ConflictingLayout,
                    detail: format!(
                        "expert {expert} has conflicting aliases {:?} and {:?}",
                        present[0], present[1]
                    ),
                    tensor_name: Some(present[1].clone()),
                    tensor_type_code: None,
                    metadata_key: None,
                });
            }
            let name = present.first().map_or(&candidates[0], |name| *name);
            validate_safetensor(store, name, &shape, false, issues);
        }
    }
    for key in keys
        .iter()
        .filter(|key| key.starts_with(&format!("{prefix}.")) && !allowed.contains(*key))
    {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: format!(
                "expert catalog {prefix:?} contains an unexpected or out-of-range tensor {key:?}"
            ),
            tensor_name: Some(key.clone()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
}

fn validate_safetensor_plan(
    store: &SafetensorsWeightStore,
    expected: Vec<ExpectedTensor>,
    quantization: Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
) -> StructuralValidation {
    validate_safetensor_format_plan(store, expected, |_| {
        quantization.map_or(
            SafetensorsMatrixFormat::Dense,
            SafetensorsMatrixFormat::Affine,
        )
    })
}

fn validate_safetensor_format_plan(
    store: &SafetensorsWeightStore,
    expected: Vec<ExpectedTensor>,
    format_for: impl Fn(&str) -> SafetensorsMatrixFormat,
) -> StructuralValidation {
    let mut common = Vec::new();
    let mut groups = Vec::new();
    let mut construction_issues = Vec::new();
    for tensor in expected {
        let format = if tensor.operation == TensorOperation::Matrix {
            format_for(&tensor.safetensors_name)
        } else {
            SafetensorsMatrixFormat::Dense
        };
        expand_safetensors_format(
            &tensor.safetensors_name,
            &tensor.safetensors_shape,
            format,
            &mut common,
            &mut groups,
            &mut construction_issues,
        );
    }
    if !construction_issues.is_empty() {
        return finish(construction_issues);
    }
    let plan = match SafetensorsCheckpointPlan::new(
        "SafeTensors",
        common,
        groups,
        CatalogPolicy::non_strict(),
    ) {
        Ok(plan) => plan,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    checkpoint_validation::validate_safetensors_plan(store, &plan)
}

fn expand_safetensors_format(
    name: &str,
    shape: &[usize],
    format: SafetensorsMatrixFormat,
    common: &mut Vec<SafetensorsTensorConstraint>,
    groups: &mut Vec<AlternativeLayoutGroup<SafetensorsTensorConstraint>>,
    issues: &mut Vec<StructuralIssue>,
) {
    match format {
        SafetensorsMatrixFormat::Dense => common.push(SafetensorsTensorConstraint::required(
            name,
            shape.to_vec(),
            StoredDtypeConstraint::Floating,
        )),
        SafetensorsMatrixFormat::Affine(quantization) => {
            let Some(&input) = shape.last() else {
                issues.push(layout(
                    name,
                    format!("quantized matrix has invalid shape {shape:?}"),
                ));
                return;
            };
            let group_size = quantization.group_size() as usize;
            let bits = quantization.bits() as usize;
            let Some(packed_input) = input.checked_mul(bits).map(|value| value / 32) else {
                issues.push(quantization_companion_issue(
                    name,
                    format!("quantized tensor {name:?} packing geometry overflows"),
                ));
                return;
            };
            if !input.is_multiple_of(group_size)
                || !input.is_multiple_of(32)
                || !input.checked_mul(bits).unwrap_or(1).is_multiple_of(32)
            {
                issues.push(quantization_companion_issue(
                    name,
                    format!(
                        "quantized tensor {name:?} input dimension {input} is incompatible with group size {group_size} and {bits}-bit packing"
                    ),
                ));
                return;
            }
            let mut packed = shape.to_vec();
            *packed.last_mut().expect("non-empty matrix shape") = packed_input;
            let packed_constraint = |key: String| {
                SafetensorsTensorConstraint::required(
                    key,
                    packed.clone(),
                    StoredDtypeConstraint::Exact(StoredDtype::U32),
                )
            };
            if let Some(alias) = quantized_weight_alias(name) {
                groups.push(AlternativeLayoutGroup {
                    id: format!("packed alias for {name}"),
                    required: true,
                    variants: vec![
                        LayoutVariant {
                            id: "canonical".into(),
                            tensors: vec![packed_constraint(name.into())],
                            discriminator_keys: vec![name.into()],
                        },
                        LayoutVariant {
                            id: "internal".into(),
                            tensors: vec![packed_constraint(alias.clone())],
                            discriminator_keys: vec![alias],
                        },
                    ],
                });
            } else {
                common.push(packed_constraint(name.into()));
            }
            let mut companion_shape = shape.to_vec();
            *companion_shape.last_mut().expect("non-empty matrix shape") = input / group_size;
            let prefix = quantized_outer_prefix(name);
            let companion_dtype = || {
                StoredDtypeConstraint::OneOf(vec![
                    StoredDtype::F16,
                    StoredDtype::BF16,
                    StoredDtype::F32,
                    StoredDtype::U8,
                ])
            };
            common.push(
                SafetensorsTensorConstraint::required(
                    format!("{prefix}.scales"),
                    companion_shape.clone(),
                    companion_dtype(),
                )
                .companion(),
            );
            if quantization.has_biases() {
                common.push(
                    SafetensorsTensorConstraint::required(
                        format!("{prefix}.biases"),
                        companion_shape,
                        companion_dtype(),
                    )
                    .companion(),
                );
            }
        }
        SafetensorsMatrixFormat::Fp8Block128 => {
            if shape.len() < 2 {
                issues.push(layout(
                    name,
                    format!("native block-FP8 weight must have rank at least two, got {shape:?}"),
                ));
                return;
            }
            common.push(SafetensorsTensorConstraint::required(
                name,
                shape.to_vec(),
                StoredDtypeConstraint::OneOf(vec![StoredDtype::F8E4M3, StoredDtype::U8]),
            ));
            let mut scale_shape = shape.to_vec();
            let rank = scale_shape.len();
            scale_shape[rank - 2] = scale_shape[rank - 2].div_ceil(128);
            scale_shape[rank - 1] = scale_shape[rank - 1].div_ceil(128);
            let scale = if name.ends_with(".weight") {
                format!("{}.weight_scale_inv", name.trim_end_matches(".weight"))
            } else {
                format!("{name}_scale_inv")
            };
            common.push(
                SafetensorsTensorConstraint::required(
                    scale,
                    scale_shape,
                    StoredDtypeConstraint::OneOf(vec![
                        StoredDtype::F16,
                        StoredDtype::BF16,
                        StoredDtype::F32,
                        StoredDtype::U8,
                    ]),
                )
                .companion(),
            );
        }
    }
}

fn add_safetensors_format_companions(
    allowed: &mut BTreeSet<String>,
    name: &str,
    format: SafetensorsMatrixFormat,
) {
    let prefix = quantized_outer_prefix(name);
    match format {
        SafetensorsMatrixFormat::Dense => {}
        SafetensorsMatrixFormat::Affine(quantization) => {
            if let Some(alias) = quantized_weight_alias(name) {
                // SafeMLX-written checkpoints expose the packed matrix through
                // the quantized module's `inner` field.  The actual loader
                // accepts that spelling as well as the canonical external
                // checkpoint spelling; companions remain on the outer module.
                allowed.insert(alias);
            }
            allowed.insert(format!("{prefix}.scales"));
            if quantization.has_biases() {
                allowed.insert(format!("{prefix}.biases"));
            }
        }
        SafetensorsMatrixFormat::Fp8Block128 => {
            let companion = if name.ends_with(".weight") {
                format!("{prefix}.weight_scale_inv")
            } else {
                format!("{name}_scale_inv")
            };
            allowed.insert(companion);
        }
    }
}

fn validate_safetensor_format(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    format: SafetensorsMatrixFormat,
    issues: &mut Vec<StructuralIssue>,
) {
    let mut common = Vec::new();
    let mut groups = Vec::new();
    expand_safetensors_format(name, shape, format, &mut common, &mut groups, issues);
    if let Ok(plan) = SafetensorsCheckpointPlan::new(
        format!("SafeTensors tensor {name}"),
        common,
        groups,
        CatalogPolicy::non_strict(),
    ) {
        append_structural_issues(
            checkpoint_validation::validate_safetensors_plan(store, &plan),
            issues,
        );
    }
}

fn validate_safetensor_plan_with(
    store: &SafetensorsWeightStore,
    expected: Vec<ExpectedTensor>,
    quantization_for: impl Fn(
        &str,
    )
        -> Option<crate::runtime::checkpoint::quantization::WeightQuantization>,
) -> StructuralValidation {
    let mut issues = Vec::new();
    for tensor in expected {
        if tensor.operation == TensorOperation::Matrix {
            if let Some(quantization) = quantization_for(&tensor.safetensors_name) {
                validate_quantized_safetensor(store, &tensor, quantization, &mut issues);
                continue;
            }
        }
        validate_safetensor(
            store,
            &tensor.safetensors_name,
            &tensor.safetensors_shape,
            false,
            &mut issues,
        );
    }
    finish(issues)
}

fn validate_quantized_safetensor(
    store: &SafetensorsWeightStore,
    tensor: &ExpectedTensor,
    quantization: crate::runtime::checkpoint::quantization::WeightQuantization,
    issues: &mut Vec<StructuralIssue>,
) {
    validate_safetensor_format(
        store,
        &tensor.safetensors_name,
        &tensor.safetensors_shape,
        SafetensorsMatrixFormat::Affine(quantization),
        issues,
    );
}

fn quantized_weight_alias(name: &str) -> Option<String> {
    if let Some(prefix) = name.strip_suffix(".inner.weight") {
        Some(format!("{prefix}.weight"))
    } else {
        name.strip_suffix(".weight")
            .map(|prefix| format!("{prefix}.inner.weight"))
    }
}

fn quantized_outer_prefix(name: &str) -> &str {
    name.strip_suffix(".inner.weight")
        .or_else(|| name.strip_suffix(".weight"))
        .unwrap_or(name)
}

fn validate_safetensor(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    packed: bool,
    issues: &mut Vec<StructuralIssue>,
) {
    let dtype = if packed {
        StoredDtypeConstraint::Exact(StoredDtype::U32)
    } else {
        StoredDtypeConstraint::Floating
    };
    let plan = SafetensorsCheckpointPlan::new(
        format!("SafeTensors tensor {name}"),
        vec![SafetensorsTensorConstraint::required(
            name,
            shape.to_vec(),
            dtype,
        )],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .expect("legacy structural tensor constraints are valid");
    append_structural_issues(
        checkpoint_validation::validate_safetensors_plan(store, &plan),
        issues,
    );
}

fn validate_quantization_companion(
    store: &SafetensorsWeightStore,
    name: &str,
    shape: &[usize],
    issues: &mut Vec<StructuralIssue>,
) {
    let plan = SafetensorsCheckpointPlan::new(
        format!("SafeTensors companion {name}"),
        vec![SafetensorsTensorConstraint::required(
            name,
            shape.to_vec(),
            StoredDtypeConstraint::OneOf(vec![
                StoredDtype::F16,
                StoredDtype::BF16,
                StoredDtype::F32,
                StoredDtype::U8,
            ]),
        )
        .companion()],
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .expect("legacy companion constraints are valid");
    append_structural_issues(
        checkpoint_validation::validate_safetensors_plan(store, &plan),
        issues,
    );
}

fn validate_deepseek2_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match deepseek_v3::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let mut expected = deepseek_v3_expected(&args);
    if args.split_kv_b {
        expected.retain(|tensor| {
            !tensor
                .safetensors_name
                .ends_with(".self_attn.kv_b_proj.weight")
        });
        for layer in 0..args.layer_schedule.len() {
            expected.extend([
                expected_rank3(
                    "",
                    format!("blk.{layer}.attn_k_b.weight"),
                    [
                        args.num_attention_heads as usize,
                        args.kv_lora_rank as usize,
                        args.qk_nope_head_dim as usize,
                    ],
                ),
                expected_rank3(
                    "",
                    format!("blk.{layer}.attn_v_b.weight"),
                    [
                        args.num_attention_heads as usize,
                        args.v_head_dim as usize,
                        args.kv_lora_rank as usize,
                    ],
                ),
            ]);
        }
    }
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if *policy != deepseek_v3::LayerPolicy::SparseMoe {
            continue;
        }
        let experts = args.n_routed_experts as usize;
        let hidden = args.hidden_size as usize;
        let intermediate = args.moe_intermediate_size as usize;
        expected.extend([
            expected_rank3(
                "",
                format!("blk.{layer}.ffn_gate_exps.weight"),
                [experts, intermediate, hidden],
            ),
            expected_rank3(
                "",
                format!("blk.{layer}.ffn_up_exps.weight"),
                [experts, intermediate, hidden],
            ),
            expected_rank3(
                "",
                format!("blk.{layer}.ffn_down_exps.weight"),
                [experts, hidden, intermediate],
            ),
        ]);
    }
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "DeepSeek2");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) && !name.starts_with("rope_freqs.") {
            issues.push(unexpected_layout(name, "DeepSeek2 GGUF"));
        }
    }
    finish(issues)
}

fn validate_deepseek4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let args = match deepseek_v4::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_nextn_predict_layers != 0 {
        return invalid_geometry(
            "canonical DeepSeek-V4 MTP GGUF files are companion artifacts and cannot be loaded as a base model".into(),
        );
    }
    let hidden = args.hidden_size as usize;
    let intermediate = args.moe_intermediate_size as usize;
    let experts = args.n_routed_experts as usize;
    let mut expected = deepseek_v4_common_expected(&args);
    for layer in 0..args.num_hidden_layers as usize {
        let root = format!("blk.{layer}");
        expected.extend([
            expected_mxfp4_rank3(
                format!("{root}.ffn_gate_exps.weight"),
                [experts, intermediate, hidden],
            ),
            expected_mxfp4_rank3(
                format!("{root}.ffn_down_exps.weight"),
                [experts, hidden, intermediate],
            ),
            expected_mxfp4_rank3(
                format!("{root}.ffn_up_exps.weight"),
                [experts, intermediate, hidden],
            ),
        ]);
    }
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "DeepSeek-V4 GGUF");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "DeepSeek-V4 GGUF"));
        }
    }
    finish(issues)
}

fn validate_kimi_linear_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(kimi_linear::translate_gguf_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let args = match kimi_linear::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = args.validate() {
        return invalid_geometry(error.to_string());
    }

    let mut issues =
        validate_gguf_plan(checkpoint, kimi_linear_expected(&args), "Kimi Linear GGUF");
    issues.extend(validate_paired_expert_encodings(
        checkpoint,
        0..args.num_hidden_layers as usize,
        "Kimi Linear GGUF",
    ));
    for (layer, policy) in args.layer_schedule.iter().enumerate() {
        if policy.attention != kimi_linear::AttentionKind::Kda {
            continue;
        }
        let projection = (args.kda_config.num_heads * args.kda_config.head_dim) as usize;
        let kernel = args.kda_config.short_conv_kernel_size as usize;
        for suffix in [
            "ssm_conv1d_q.weight",
            "ssm_conv1d_k.weight",
            "ssm_conv1d_v.weight",
        ] {
            issues.extend(validate_gguf_element_count(
                checkpoint,
                &format!("blk.{layer}.{suffix}"),
                projection * kernel,
                TensorOperation::Dense,
                "Kimi Linear GGUF",
            ));
        }
        let canonical = format!("blk.{layer}.ssm_a");
        let weight_alias = format!("{canonical}.weight");
        let name = if checkpoint
            .catalog()
            .tensors()
            .any(|tensor| tensor.descriptor().name == weight_alias)
        {
            weight_alias
        } else {
            canonical
        };
        issues.extend(validate_gguf_element_count(
            checkpoint,
            &name,
            args.kda_config.num_heads as usize,
            TensorOperation::Vector,
            "Kimi Linear GGUF",
        ));
    }
    finish(issues)
}

fn validate_gguf_element_count(
    checkpoint: &GgufCheckpoint,
    name: &str,
    expected_elements: usize,
    operation: TensorOperation,
    loader_name: &str,
) -> Vec<StructuralIssue> {
    let Some(actual) = checkpoint
        .catalog()
        .tensors()
        .find(|tensor| tensor.descriptor().name == name)
    else {
        return vec![missing(name)];
    };
    let mut issues = Vec::new();
    let elements = actual
        .descriptor()
        .mlx_shape()
        .into_iter()
        .try_fold(1usize, |product, dimension| {
            product.checked_mul(usize::try_from(dimension).ok()?)
        });
    if elements != Some(expected_elements) {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ShapeMismatch,
            detail: format!(
                "tensor {name:?} must contain {expected_elements} elements for the loader transform, got {:?}",
                actual.descriptor().mlx_shape()
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: Some(actual.descriptor().ggml_type.code()),
            metadata_key: None,
        });
    }
    if !gguf_encoding_supported(operation, actual.descriptor().ggml_type) {
        let encoding = actual.descriptor().ggml_type;
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::UnsupportedEncoding,
            detail: format!(
                "GGUF tensor {name:?} uses {encoding:?} (type {}) for a {operation:?} operation, which the {loader_name} loader does not support",
                encoding.code()
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: Some(encoding.code()),
            metadata_key: None,
        });
    }
    issues
}

fn validate_lfm2_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let is_moe = architecture == GgufArchitecture::Lfm2Moe;
    let metadata_name = architecture.metadata_name();
    let translate = |name: &str| lfm2::translate_gguf_weight_name(name, is_moe);
    if let Err(error) = checkpoint.catalog().translated_outputs(translate) {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match lfm2::args_from_gguf_catalog(checkpoint, metadata, metadata_name, is_moe) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let mut expected = match lfm2_expected(&args) {
        Ok(expected) => expected,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    for tensor in &mut expected {
        if let Some(prefix) = tensor.gguf_name.strip_suffix(".ffn_exp_probs_b.bias") {
            let alias = format!("{prefix}.exp_probs_b.bias");
            if checkpoint
                .catalog()
                .tensors()
                .any(|actual| actual.descriptor().name == alias)
            {
                tensor.gguf_name = alias;
            }
        }
    }
    let mut issues = validate_gguf_plan(checkpoint, expected, "LFM2");
    if is_moe {
        issues.extend(validate_paired_expert_encodings(
            checkpoint,
            args.layer_schedule
                .iter()
                .enumerate()
                .filter_map(|(layer, policy)| {
                    (policy.feed_forward == lfm2::FeedForwardPolicy::SparseMoe).then_some(layer)
                }),
            "LFM2 MoE",
        ));
    }
    finish(issues)
}

fn inkling_gguf_alias(checkpoint: &GgufCheckpoint, names: &[String]) -> String {
    names
        .iter()
        .find(|name| {
            checkpoint
                .catalog()
                .tensors()
                .any(|tensor| tensor.descriptor().name == **name)
        })
        .cloned()
        .unwrap_or_else(|| names[0].clone())
}

fn inkling_gguf_expected(
    args: &inkling::ModelArgs,
    checkpoint: &GgufCheckpoint,
    flexible: &mut Vec<(String, Vec<Vec<usize>>, TensorOperation)>,
) -> Vec<ExpectedTensor> {
    let text = &args.text_config;
    let hidden = text.hidden_size as usize;
    let vocab = text.vocab_size as usize;
    let mut tensors = vec![
        expected("", "token_embd.weight", [vocab, hidden]),
        expected_vector_shape("token_embd_norm.weight", vec![hidden]),
        expected_vector_shape("output_norm.weight", vec![hidden]),
        expected("", "output.weight", [vocab, hidden]),
    ];
    for layer in 0..text.num_hidden_layers {
        let prefix = format!("blk.{layer}");
        let policy = *text
            .layer_policy(layer as usize)
            .expect("validated Inkling layer schedule");
        let local = policy.attention.window().is_some();
        let query_heads = text.q_heads(local) as usize;
        let kv_heads = text.kv_heads(local) as usize;
        let head = text.attention_head_dim(local) as usize;
        let relative = policy
            .attention
            .window()
            .map(|window| window.get() as usize)
            .unwrap_or(text.rel_extent as usize);
        tensors.extend([
            expected_vector_shape(format!("{prefix}.attn_norm.weight"), vec![hidden]),
            expected_vector_shape(format!("{prefix}.ffn_norm.weight"), vec![hidden]),
            expected(
                "",
                format!("{prefix}.attn_q.weight"),
                [query_heads * head, hidden],
            ),
            expected(
                "",
                format!("{prefix}.attn_k.weight"),
                [kv_heads * head, hidden],
            ),
            expected(
                "",
                format!("{prefix}.attn_v.weight"),
                [kv_heads * head, hidden],
            ),
            expected(
                "",
                format!("{prefix}.attn_r.weight"),
                [query_heads * text.d_rel as usize, hidden],
            ),
            expected(
                "",
                format!("{prefix}.attn_output.weight"),
                [hidden, query_heads * head],
            ),
            expected_vector_shape(format!("{prefix}.attn_q_norm.weight"), vec![head]),
            expected_vector_shape(format!("{prefix}.attn_k_norm.weight"), vec![head]),
        ]);
        let relative_name = inkling_gguf_alias(
            checkpoint,
            &[
                format!("{prefix}.attn_rel_proj.weight"),
                format!("{prefix}.attn_rel_proj"),
            ],
        );
        tensors.push(expected_dense_with_gguf_shape(
            "",
            relative_name,
            [text.d_rel as usize, relative],
            [text.d_rel as usize, relative],
        ));
        for (name, channels) in [
            ("shortconv_k", kv_heads * head),
            ("shortconv_v", kv_heads * head),
            ("shortconv_attn", hidden),
            ("shortconv_mlp", hidden),
        ] {
            flexible.push((
                format!("{prefix}.{name}.weight"),
                vec![
                    vec![channels, text.sconv_kernel_size as usize],
                    vec![channels, 1, text.sconv_kernel_size as usize],
                ],
                TensorOperation::Dense,
            ));
        }

        if policy.feed_forward == inkling::FeedForwardPolicy::Dense {
            let intermediate = text.dense_intermediate_size() as usize;
            tensors.extend([
                expected(
                    "",
                    format!("{prefix}.ffn_gate.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    "",
                    format!("{prefix}.ffn_up.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    "",
                    format!("{prefix}.ffn_down.weight"),
                    [hidden, intermediate],
                ),
            ]);
            let scale = inkling_gguf_alias(
                checkpoint,
                &[
                    format!("{prefix}.ffn_gscale"),
                    format!("{prefix}.ffn_gscale.weight"),
                ],
            );
            tensors.push(expected_vector_shape(scale, vec![1]));
        } else {
            let routed = text.n_routed_experts as usize;
            let shared = text.n_shared_experts as usize;
            let intermediate = text.moe_intermediate_size() as usize;
            tensors.push(expected_dense_with_gguf_shape(
                "",
                format!("{prefix}.ffn_gate_inp.weight"),
                [routed + shared, hidden],
                [routed + shared, hidden],
            ));
            let bias = inkling_gguf_alias(
                checkpoint,
                &[
                    format!("{prefix}.exp_probs_b.bias"),
                    format!("{prefix}.ffn_exp_probs_b.bias"),
                    format!("{prefix}.ffn_exp_probs_b"),
                ],
            );
            let scale = inkling_gguf_alias(
                checkpoint,
                &[
                    format!("{prefix}.ffn_gscale"),
                    format!("{prefix}.ffn_gscale.weight"),
                ],
            );
            tensors.extend([
                expected_vector_shape(bias, vec![routed]),
                expected_vector_shape(scale, vec![1]),
                expected_rank3(
                    "",
                    format!("{prefix}.ffn_gate_exps.weight"),
                    [routed, intermediate, hidden],
                ),
                expected_rank3(
                    "",
                    format!("{prefix}.ffn_up_exps.weight"),
                    [routed, intermediate, hidden],
                ),
                expected_rank3(
                    "",
                    format!("{prefix}.ffn_down_exps.weight"),
                    [routed, hidden, intermediate],
                ),
                expected_rank3(
                    "",
                    format!("{prefix}.ffn_gate_shexp.weight"),
                    [shared, intermediate, hidden],
                ),
                expected_rank3(
                    "",
                    format!("{prefix}.ffn_up_shexp.weight"),
                    [shared, intermediate, hidden],
                ),
                expected_rank3(
                    "",
                    format!("{prefix}.ffn_down_shexp.weight"),
                    [shared, hidden, intermediate],
                ),
            ]);
        }
    }
    tensors
}

fn validate_inkling_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = GgufArchitecture::Inkling.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(inkling::translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match inkling::args_from_gguf_catalog(metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.text_config.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor Inkling GGUF catalog",
            args.text_config.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let mut flexible = Vec::new();
    let expected = inkling_gguf_expected(&args, checkpoint, &mut flexible);
    let mut allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Inkling");
    for (name, shapes, operation) in flexible {
        allowed.insert(name.clone());
        issues.extend(validate_gguf_one_of(
            checkpoint, &name, &shapes, operation, "Inkling",
        ));
    }
    for layer in args
        .text_config
        .layer_schedule
        .iter()
        .enumerate()
        .filter_map(|(layer, policy)| {
            (policy.feed_forward == inkling::FeedForwardPolicy::SparseMoe).then_some(layer)
        })
    {
        for (gate, up) in [
            (
                format!("blk.{layer}.ffn_gate_exps.weight"),
                format!("blk.{layer}.ffn_up_exps.weight"),
            ),
            (
                format!("blk.{layer}.ffn_gate_shexp.weight"),
                format!("blk.{layer}.ffn_up_shexp.weight"),
            ),
        ] {
            issues.extend(validate_inkling_paired_gguf_formats(checkpoint, &gate, &up));
        }
    }
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Inkling GGUF"));
        }
    }
    finish(issues)
}

pub(crate) fn validate_inkling_mmproj_gguf(
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &inkling::InklingMmprojGguf,
) -> StructuralValidation {
    if let Err(error) = mmproj
        .checkpoint
        .catalog()
        .translated_outputs(inkling::translate_mmproj_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let mut args = match inkling::args_from_gguf_catalog(model_metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if let Err(error) = inkling::apply_mmproj_args(&mut args, model_metadata, mmproj) {
        return invalid_geometry(error.to_string());
    }
    let audio = args
        .audio_config
        .as_ref()
        .expect("validated Inkling audio projector");
    let vision = args
        .vision_config
        .as_ref()
        .expect("validated Inkling vision projector");
    let hidden = args.text_config.hidden_size as usize;
    let mut plan = vec![
        expected(
            "",
            "a.dmel.embedding.weight",
            [(audio.num_codebooks * audio.codebook_size) as usize, hidden],
        ),
        expected_vector_shape("a.dmel.final_norm.weight", vec![hidden]),
    ];
    for (layer, (input, output, _, _)) in vision.layer_specs().into_iter().enumerate() {
        plan.push(expected(
            "",
            format!("v.hmlp.{layer}.linear.weight"),
            [output as usize, input as usize],
        ));
        if layer + 1 != vision.layer_specs().len() {
            plan.push(expected_vector_shape(
                format!("v.hmlp.{layer}.norm.weight"),
                vec![output as usize],
            ));
        }
    }
    plan.push(expected_vector_shape(
        "v.hmlp.final_norm.weight",
        vec![hidden],
    ));
    let allowed = plan
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(&mmproj.checkpoint, plan, "Inkling mmproj");
    for tensor in mmproj.checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Inkling mmproj GGUF"));
        }
    }
    finish(issues)
}

pub(crate) fn validate_gemma4_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &gemma4::Gemma4MmprojGguf,
) -> StructuralValidation {
    if let Err(error) = gemma4::validate_mmproj_metadata(&mmproj.metadata) {
        return invalid_geometry(error.to_string());
    }
    if let Err(error) = mmproj
        .checkpoint
        .catalog()
        .translated_outputs(gemma4::translate_mmproj_weight_name)
    {
        return invalid_geometry(error.to_string());
    }
    let mut args = match gemma4::gemma4_args_from_gguf_catalog(model_checkpoint, model_metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let (vision, _, _, audio, _) =
        match gemma4::apply_mmproj_args(&mut args, model_metadata, mmproj) {
            Ok(parts) => parts,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    let plan = std::cell::RefCell::new(Vec::new());
    let push = |physical: String, shape: Vec<usize>| {
        let logical = gemma4::translate_mmproj_weight_name(&physical);
        plan.borrow_mut().push(expected_dense_with_gguf_shape(
            logical,
            physical,
            shape.clone(),
            shape,
        ));
    };
    let clipped = |prefix: String, shape: Vec<usize>| {
        push(format!("{prefix}.linear.weight"), shape);
        for suffix in ["input_min", "input_max", "output_min", "output_max"] {
            push(format!("{prefix}.{suffix}"), vec![]);
        }
    };
    if let Some(config) = vision {
        let hidden = config.hidden_size as usize;
        let intermediate = config.intermediate_size as usize;
        let query = config.num_attention_heads as usize * config.head_dim as usize;
        let key_value = config.num_key_value_heads as usize * config.head_dim as usize;
        push(
            "vision_tower.patch_embedder.input_proj.weight".into(),
            vec![hidden, 3 * (config.patch_size as usize).pow(2)],
        );
        push(
            "vision_tower.patch_embedder.position_embedding_table".into(),
            vec![2, config.position_embedding_size as usize, hidden],
        );
        if config.standardize {
            push("vision_tower.std_bias".into(), vec![hidden]);
            push("vision_tower.std_scale".into(), vec![hidden]);
        }
        for layer in 0..config.num_hidden_layers as usize {
            let root = format!("vision_tower.encoder.layers.{layer}");
            for name in [
                "input_layernorm.weight",
                "post_attention_layernorm.weight",
                "pre_feedforward_layernorm.weight",
                "post_feedforward_layernorm.weight",
                "self_attn.q_norm.weight",
                "self_attn.k_norm.weight",
            ] {
                let width = if name.contains("q_norm") || name.contains("k_norm") {
                    config.head_dim as usize
                } else {
                    hidden
                };
                push(format!("{root}.{name}"), vec![width]);
            }
            for (name, shape) in [
                ("self_attn.q_proj", vec![query, hidden]),
                ("self_attn.k_proj", vec![key_value, hidden]),
                ("self_attn.v_proj", vec![key_value, hidden]),
                ("self_attn.o_proj", vec![hidden, query]),
                ("mlp.gate_proj", vec![intermediate, hidden]),
                ("mlp.up_proj", vec![intermediate, hidden]),
                ("mlp.down_proj", vec![hidden, intermediate]),
            ] {
                clipped(format!("{root}.{name}"), shape);
            }
        }
        push(
            "embed_vision.embedding_projection.weight".into(),
            vec![args.hidden_size as usize, hidden],
        );
    }
    if let Some(config) = audio {
        let hidden = config.hidden_size as usize;
        let head = hidden / config.num_attention_heads as usize;
        let [first, second] = config.subsampling_conv_channels.as_slice() else {
            return invalid_geometry(
                "Gemma 4 audio requires exactly two subsampling convolution channels".into(),
            );
        };
        let first = *first as usize;
        let second = *second as usize;
        for (name, shape) in [
            (
                "audio_tower.subsample_conv_projection.layer0.conv.weight",
                vec![first, 3, 3, 1],
            ),
            (
                "audio_tower.subsample_conv_projection.layer0.norm.weight",
                vec![first],
            ),
            (
                "audio_tower.subsample_conv_projection.layer1.conv.weight",
                vec![second, 3, 3, first],
            ),
            (
                "audio_tower.subsample_conv_projection.layer1.norm.weight",
                vec![second],
            ),
            (
                "audio_tower.subsample_conv_projection.input_proj_linear.weight",
                vec![hidden, 32 * second],
            ),
            (
                "audio_tower.output_proj.weight",
                vec![config.output_proj_dims as usize, hidden],
            ),
            (
                "audio_tower.output_proj.bias",
                vec![config.output_proj_dims as usize],
            ),
        ] {
            push(name.into(), shape);
        }
        for layer in 0..config.num_hidden_layers as usize {
            let root = format!("audio_tower.layers.{layer}");
            for name in [
                "feed_forward1.pre_layer_norm.weight",
                "feed_forward1.post_layer_norm.weight",
                "norm_pre_attn.weight",
                "norm_post_attn.weight",
                "lconv1d.pre_layer_norm.weight",
                "lconv1d.conv_norm.weight",
                "feed_forward2.pre_layer_norm.weight",
                "feed_forward2.post_layer_norm.weight",
                "norm_out.weight",
            ] {
                push(format!("{root}.{name}"), vec![hidden]);
            }
            for (name, shape) in [
                ("feed_forward1.ffw_layer_1", vec![4 * hidden, hidden]),
                ("feed_forward1.ffw_layer_2", vec![hidden, 4 * hidden]),
                ("self_attn.q_proj", vec![hidden, hidden]),
                ("self_attn.k_proj", vec![hidden, hidden]),
                ("self_attn.v_proj", vec![hidden, hidden]),
                ("self_attn.post", vec![hidden, hidden]),
                ("lconv1d.linear_start", vec![2 * hidden, hidden]),
                ("lconv1d.linear_end", vec![hidden, hidden]),
                ("feed_forward2.ffw_layer_1", vec![4 * hidden, hidden]),
                ("feed_forward2.ffw_layer_2", vec![hidden, 4 * hidden]),
            ] {
                clipped(format!("{root}.{name}"), shape);
            }
            for (name, shape) in [
                ("self_attn.relative_k_proj.weight", vec![hidden, hidden]),
                ("self_attn.per_dim_scale", vec![head]),
                (
                    "lconv1d.depthwise_conv1d.weight",
                    vec![hidden, config.conv_kernel_size as usize, 1],
                ),
            ] {
                push(format!("{root}.{name}"), shape);
            }
        }
        push(
            "embed_audio.embedding_projection.weight".into(),
            vec![args.hidden_size as usize, config.output_proj_dims as usize],
        );
    }
    let plan = plan.into_inner();
    let allowed = plan
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(&mmproj.checkpoint, plan, "Gemma 4 mmproj");
    for tensor in mmproj.checkpoint.catalog().tensors() {
        let descriptor = tensor.descriptor();
        if !allowed.contains(&descriptor.name) {
            issues.push(unexpected_layout(&descriptor.name, "Gemma 4 mmproj GGUF"));
        } else if !matches!(
            descriptor.ggml_type,
            GgufType::F32 | GgufType::F16 | GgufType::Bf16
        ) {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::UnsupportedEncoding,
                detail: format!(
                    "Gemma 4 mmproj tensor {:?} must be dense F16, BF16, or F32, got {:?}",
                    descriptor.name, descriptor.ggml_type
                ),
                tensor_name: Some(descriptor.name.clone()),
                tensor_type_code: Some(descriptor.ggml_type.code()),
                metadata_key: None,
            });
        }
    }
    finish(issues)
}

fn gemma4_gguf_expected(
    args: &gemma4::ModelArgs,
    checkpoint: &GgufCheckpoint,
    issues: &mut Vec<StructuralIssue>,
) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let layers = args.num_hidden_layers as usize;
    let vocab = args.vocab_size as usize;
    let mut tensors = vec![
        expected(
            "model.language_model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector(
            "model.language_model.norm.weight",
            "output_norm.weight",
            hidden,
        ),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }
    if args.hidden_size_per_layer_input > 0 {
        let per_layer = args.hidden_size_per_layer_input as usize;
        let combined = layers * per_layer;
        let per_layer_vocab = args.vocab_size_per_layer_input.unwrap_or(args.vocab_size) as usize;
        tensors.extend([
            expected(
                "model.language_model.embed_tokens_per_layer.weight",
                "per_layer_token_embd.weight",
                [per_layer_vocab, combined],
            ),
            expected(
                "model.language_model.per_layer_model_projection.weight",
                "per_layer_model_proj.weight",
                [combined, hidden],
            ),
            expected_vector(
                "model.language_model.per_layer_projection_norm.weight",
                "per_layer_proj_norm.weight",
                per_layer,
            ),
        ]);
    }

    let catalog = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| tensor.descriptor().name.as_str())
        .collect::<BTreeSet<_>>();
    for layer in 0..layers {
        let gguf = format!("blk.{layer}");
        let policy = args
            .layer_policy(layer)
            .expect("validated Gemma 4 layer schedule");
        let head_dim = policy.head_dim.get() as usize;
        let kv_heads = policy.num_key_value_heads.get() as usize;
        let query = args.num_attention_heads as usize * head_dim;
        let key_value = kv_heads * head_dim;
        let shared_kv = !policy.key_value.owns_state();
        let attention_k_eq_v = policy.key_value.value() == Some(gemma4::ValuePolicy::ReuseKey);
        let intermediate = policy.intermediate_size.get() as usize;

        for name in [
            "attn_norm",
            "post_attention_norm",
            "ffn_norm",
            "post_ffw_norm",
        ] {
            tensors.push(expected_vector_shape(
                format!("{gguf}.{name}.weight"),
                vec![hidden],
            ));
        }
        tensors.extend([
            expected_vector_shape(format!("{gguf}.layer_output_scale.weight"), vec![1]),
            expected("", format!("{gguf}.attn_q.weight"), [query, hidden]),
            expected("", format!("{gguf}.attn_output.weight"), [hidden, query]),
            expected_vector_shape(format!("{gguf}.attn_q_norm.weight"), vec![head_dim]),
            expected(
                "",
                format!("{gguf}.ffn_gate.weight"),
                [intermediate, hidden],
            ),
            expected("", format!("{gguf}.ffn_up.weight"), [intermediate, hidden]),
            expected(
                "",
                format!("{gguf}.ffn_down.weight"),
                [hidden, intermediate],
            ),
        ]);
        if !shared_kv {
            tensors.extend([
                expected("", format!("{gguf}.attn_k.weight"), [key_value, hidden]),
                expected_vector_shape(format!("{gguf}.attn_k_norm.weight"), vec![head_dim]),
            ]);
            if !attention_k_eq_v {
                tensors.push(expected(
                    "",
                    format!("{gguf}.attn_v.weight"),
                    [key_value, hidden],
                ));
            }
        }
        if args.hidden_size_per_layer_input > 0 {
            let per_layer = args.hidden_size_per_layer_input as usize;
            tensors.extend([
                expected("", format!("{gguf}.inp_gate.weight"), [per_layer, hidden]),
                expected("", format!("{gguf}.proj.weight"), [hidden, per_layer]),
                expected_vector_shape(format!("{gguf}.post_norm.weight"), vec![hidden]),
            ]);
        }
        if policy.feed_forward == gemma4::FeedForwardPolicy::DenseWithSparseMoe {
            let experts = args.num_experts.expect("validated Gemma 4 MoE") as usize;
            let moe = args.moe_intermediate_size.expect("validated Gemma 4 MoE") as usize;
            tensors.extend([
                expected("", format!("{gguf}.ffn_gate_inp.weight"), [experts, hidden]),
                expected_vector_shape(format!("{gguf}.ffn_gate_inp.scale"), vec![hidden]),
                expected_vector_shape(format!("{gguf}.ffn_down_exps.scale"), vec![experts]),
                expected_vector_shape(format!("{gguf}.post_ffw_norm_1.weight"), vec![hidden]),
                expected_vector_shape(format!("{gguf}.pre_ffw_norm_2.weight"), vec![hidden]),
                expected_vector_shape(format!("{gguf}.post_ffw_norm_2.weight"), vec![hidden]),
                expected_rank3(
                    "",
                    format!("{gguf}.ffn_down_exps.weight"),
                    [experts, hidden, moe],
                ),
            ]);
            let fused_name = format!("{gguf}.ffn_gate_up_exps.weight");
            let gate_name = format!("{gguf}.ffn_gate_exps.weight");
            let up_name = format!("{gguf}.ffn_up_exps.weight");
            let fused = catalog.contains(fused_name.as_str());
            let gate = catalog.contains(gate_name.as_str());
            let up = catalog.contains(up_name.as_str());
            if fused && (gate || up) {
                issues.push(StructuralIssue {
                    kind: StructuralIssueKind::ConflictingLayout,
                    detail: format!(
                        "Gemma 4 GGUF layer {layer} mixes fused and separate gate/up expert tensors"
                    ),
                    tensor_name: Some(fused_name.clone()),
                    tensor_type_code: None,
                    metadata_key: None,
                });
            }
            if fused {
                tensors.push(expected_rank3("", fused_name, [experts, 2 * moe, hidden]));
            } else {
                tensors.extend([
                    expected_rank3("", gate_name, [experts, moe, hidden]),
                    expected_rank3("", up_name, [experts, moe, hidden]),
                ]);
            }
        }
    }
    tensors
}

fn validate_gemma4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = GgufArchitecture::Gemma4.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(gemma4::translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match gemma4::gemma4_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let mut issues = Vec::new();
    let expected = gemma4_gguf_expected(&args, checkpoint, &mut issues);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    issues.extend(validate_gguf_plan(checkpoint, expected, "Gemma 4"));
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        let optional_shared_kv = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split_once('.'))
            .and_then(|(layer, parameter)| {
                layer.parse::<usize>().ok().map(|layer| (layer, parameter))
            })
            .is_some_and(|(layer, parameter)| {
                args.layer_policy(layer)
                    .is_some_and(|policy| !policy.key_value.owns_state())
                    && ["attn_k.", "attn_v.", "attn_k_norm."]
                        .iter()
                        .any(|prefix| parameter.starts_with(prefix))
            });
        if !allowed.contains(name) && !name.starts_with("rope_freqs.") && !optional_shared_kv {
            issues.push(unexpected_layout(name, "Gemma 4 GGUF"));
        }
    }
    finish(issues)
}

fn validate_gpt_oss_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(gpt_oss::translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match gpt_oss::args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let mut expected = gpt_oss_gguf_expected(&args);
    for tensor in &mut expected {
        if let Some(alias) = tensor.gguf_name.strip_suffix(".attn_sinks.weight") {
            let alias = format!("{alias}.attn_sinks");
            if checkpoint
                .catalog()
                .tensors()
                .any(|actual| actual.descriptor().name == alias)
            {
                tensor.gguf_name = alias;
            }
        }
    }
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "GPT-OSS");
    for actual in checkpoint.catalog().tensors() {
        if !allowed.contains(&actual.descriptor().name) {
            issues.push(unexpected_layout(&actual.descriptor().name, "GPT-OSS GGUF"));
        }
    }
    issues.extend(validate_paired_expert_encodings(
        checkpoint,
        0..args.num_hidden_layers as usize,
        "GPT-OSS",
    ));
    finish(issues)
}

fn validate_llama_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(llama::translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match llama::model_args_from_gguf_catalog(checkpoint, metadata) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let expected = llama_expected(&args);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Llama");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) && !name.starts_with("rope_freqs.") {
            issues.push(unexpected_layout(name, "Llama GGUF"));
        }
    }
    finish(issues)
}

fn nemotron_h_gguf_expected(args: &nemotron_h::ModelArgs) -> Result<Vec<ExpectedTensor>, Error> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let mut tensors = vec![
        expected(
            "model.embeddings.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm_f.weight", "output_norm.weight", hidden),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }

    for (layer, block_type) in args.layer_schedule.iter().copied().enumerate() {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.push(expected_vector(
            format!("{model}.norm.weight"),
            format!("{gguf}.attn_norm.weight"),
            hidden,
        ));
        match block_type {
            nemotron_h::LayerPolicy::Mamba => {
                let heads = args.mamba_num_heads as usize;
                let intermediate = heads * args.mamba_head_dim as usize;
                let conv = intermediate + 2 * args.n_groups as usize * args.ssm_state_size as usize;
                let projection = intermediate + conv + heads;
                tensors.extend([
                    expected(
                        format!("{model}.mamba.in_proj.weight"),
                        format!("{gguf}.ssm_in.weight"),
                        [projection, hidden],
                    ),
                    expected_dense_with_gguf_shape(
                        format!("{model}.mamba.conv1d.weight"),
                        format!("{gguf}.ssm_conv1d.weight"),
                        vec![conv, 1, args.conv_kernel as usize],
                        vec![conv, args.conv_kernel as usize],
                    ),
                    expected_vector_shape(format!("{gguf}.ssm_dt.bias"), vec![heads]),
                    ExpectedTensor {
                        safetensors_name: format!("{model}.mamba.A_log"),
                        gguf_name: format!("{gguf}.ssm_a"),
                        safetensors_shape: vec![heads],
                        gguf_shape: vec![heads],
                        operation: TensorOperation::Dense,
                    },
                    expected_vector_shape(format!("{gguf}.ssm_d"), vec![heads]),
                    expected_vector(
                        format!("{model}.mamba.norm.weight"),
                        format!("{gguf}.ssm_norm.weight"),
                        intermediate,
                    ),
                    expected(
                        format!("{model}.mamba.out_proj.weight"),
                        format!("{gguf}.ssm_out.weight"),
                        [hidden, intermediate],
                    ),
                ]);
                if args.use_conv_bias {
                    tensors.push(expected_vector_shape(
                        format!("{gguf}.ssm_conv1d.bias"),
                        vec![conv],
                    ));
                }
                if args.use_bias {
                    tensors.extend([
                        expected_vector_shape(format!("{gguf}.ssm_in.bias"), vec![projection]),
                        expected_vector_shape(format!("{gguf}.ssm_out.bias"), vec![hidden]),
                    ]);
                }
            }
            nemotron_h::LayerPolicy::SelfAttention(_) => {
                let query = (args.num_attention_heads * args.head_dim) as usize;
                let key_value = (args.num_key_value_heads * args.head_dim) as usize;
                for (name, output, input) in [
                    ("attn_q", query, hidden),
                    ("attn_k", key_value, hidden),
                    ("attn_v", key_value, hidden),
                    ("attn_output", hidden, query),
                ] {
                    tensors.push(expected(
                        "",
                        format!("{gguf}.{name}.weight"),
                        [output, input],
                    ));
                    if args.attention_bias {
                        tensors.push(expected_vector_shape(
                            format!("{gguf}.{name}.bias"),
                            vec![output],
                        ));
                    }
                }
            }
            nemotron_h::LayerPolicy::DenseMlp => {
                let intermediate = args.intermediate_size as usize;
                tensors.extend([
                    expected("", format!("{gguf}.ffn_up.weight"), [intermediate, hidden]),
                    expected(
                        "",
                        format!("{gguf}.ffn_down.weight"),
                        [hidden, intermediate],
                    ),
                ]);
                if args.mlp_bias {
                    tensors.extend([
                        expected_vector_shape(format!("{gguf}.ffn_up.bias"), vec![intermediate]),
                        expected_vector_shape(format!("{gguf}.ffn_down.bias"), vec![hidden]),
                    ]);
                }
            }
            nemotron_h::LayerPolicy::SparseMoe => {
                let experts = args.n_routed_experts as usize;
                let intermediate = args.moe_intermediate_size as usize;
                let shared = args.moe_shared_expert_intermediate_size as usize;
                tensors.extend([
                    expected("", format!("{gguf}.ffn_gate_inp.weight"), [experts, hidden]),
                    expected_vector_shape(format!("{gguf}.exp_probs_b.bias"), vec![experts]),
                    expected("", format!("{gguf}.ffn_up_shexp.weight"), [shared, hidden]),
                    expected(
                        "",
                        format!("{gguf}.ffn_down_shexp.weight"),
                        [hidden, shared],
                    ),
                    expected_rank3(
                        "",
                        format!("{gguf}.ffn_up_exps.weight"),
                        [experts, intermediate, hidden],
                    ),
                    expected_rank3(
                        "",
                        format!("{gguf}.ffn_down_exps.weight"),
                        [experts, hidden, intermediate],
                    ),
                ]);
                if args.mlp_bias {
                    tensors.extend([
                        expected_vector_shape(format!("{gguf}.ffn_up_shexp.bias"), vec![shared]),
                        expected_vector_shape(format!("{gguf}.ffn_down_shexp.bias"), vec![hidden]),
                    ]);
                }
            }
        }
    }
    Ok(tensors)
}

fn validate_nemotron_h_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = architecture.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(nemotron_h::translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match nemotron_h::model_args_from_gguf_catalog(
        checkpoint,
        metadata,
        architecture.metadata_name(),
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let mut expected = match nemotron_h_gguf_expected(&args) {
        Ok(expected) => expected,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    for tensor in &mut expected {
        if let Some(prefix) = tensor.gguf_name.strip_suffix(".exp_probs_b.bias") {
            let alias = format!("{prefix}.ffn_exp_probs_b.bias");
            if checkpoint
                .catalog()
                .tensors()
                .any(|actual| actual.descriptor().name == alias)
            {
                tensor.gguf_name = alias;
            }
        }
    }
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Nemotron-H");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) && !name.starts_with("rope_freqs.") {
            issues.push(unexpected_layout(name, "Nemotron-H GGUF"));
        }
    }
    finish(issues)
}

fn validate_dense_qwen_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let is_moe = architecture == GgufArchitecture::Qwen3Moe;
    let metadata_name = architecture.metadata_name();
    let translate = |name: &str| dense_qwen::translate_gguf_weight_name(name, is_moe);
    if let Err(error) = checkpoint.catalog().translated_outputs(translate) {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args =
        match dense_qwen::config_from_gguf_catalog(checkpoint, metadata, metadata_name, is_moe) {
            Ok(args) => args,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    if args.num_hidden_layers as usize > checkpoint.catalog().physical_tensor_count() {
        return invalid_geometry(format!(
            "configured layer count {} exceeds the entire {}-tensor GGUF catalog",
            args.num_hidden_layers,
            checkpoint.catalog().physical_tensor_count()
        ));
    }
    let mut issues = validate_gguf_plan(
        checkpoint,
        dense_qwen_expected(&args),
        if architecture == GgufArchitecture::Qwen2 {
            "Qwen2"
        } else {
            "Qwen3"
        },
    );
    if is_moe {
        issues.extend(validate_paired_expert_encodings(
            checkpoint,
            0..args.num_hidden_layers as usize,
            "Qwen3 MoE",
        ));
    }
    finish(issues)
}

fn muse_glimmer_gguf_expected(args: &muse_glimmer::DecoderConfig) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let query = (args.num_attention_heads * args.head_dim) as usize;
    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
    let head = args.head_dim as usize;
    let intermediate = args.intermediate_size as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
        expected("lm_head.weight", "output.weight", [vocab, hidden]),
    ];
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.input_layernorm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_attention_layernorm.weight"),
                format!("{gguf}.post_attention_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.pre_feedforward_layernorm.weight"),
                format!("{gguf}.ffn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_feedforward_layernorm.weight"),
                format!("{gguf}.post_ffw_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.self_attn.q_norm.weight"),
                format!("{gguf}.attn_q_norm.weight"),
                head,
            ),
            expected_vector(
                format!("{model}.self_attn.k_norm.weight"),
                format!("{gguf}.attn_k_norm.weight"),
                head,
            ),
            expected(
                format!("{model}.self_attn.q_proj.weight"),
                format!("{gguf}.attn_q.weight"),
                [query, hidden],
            ),
            expected(
                format!("{model}.self_attn.k_proj.weight"),
                format!("{gguf}.attn_k.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.v_proj.weight"),
                format!("{gguf}.attn_v.weight"),
                [key_value, hidden],
            ),
            expected(
                format!("{model}.self_attn.o_proj.weight"),
                format!("{gguf}.attn_output.weight"),
                [hidden, query],
            ),
            expected(
                format!("{model}.self_attn.gate_proj.weight"),
                format!("{gguf}.attn_gate.weight"),
                [query, hidden],
            ),
            expected(
                format!("{model}.mlp.gate_proj.weight"),
                format!("{gguf}.ffn_gate.weight"),
                [intermediate, hidden],
            ),
            expected(
                format!("{model}.mlp.up_proj.weight"),
                format!("{gguf}.ffn_up.weight"),
                [intermediate, hidden],
            ),
            expected(
                format!("{model}.mlp.down_proj.weight"),
                format!("{gguf}.ffn_down.weight"),
                [hidden, intermediate],
            ),
        ]);
    }
    tensors
}

fn validate_muse_glimmer_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let translate = |name: &str| muse_glimmer::translate_gguf_weight_name(name, false);
    if let Err(error) = checkpoint.catalog().translated_outputs(translate) {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args =
        match muse_glimmer::config_from_gguf_catalog(checkpoint, metadata, "muse-glimmer", false) {
            Ok(args) => args,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    let expected = muse_glimmer_gguf_expected(&args);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Muse-Glimmer");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Muse-Glimmer GGUF"));
        }
    }
    finish(issues)
}

pub(crate) fn validate_muse_glimmer_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let text = match muse_glimmer::config_from_gguf_catalog(
        model_checkpoint,
        model_metadata,
        "muse-glimmer",
        false,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let vision =
        match muse_glimmer::vision::VisionConfig::from_gguf_metadata(metadata, text.hidden_size) {
            Ok(config) => config,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(muse_glimmer::translate_mmproj_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let dense = |name: String, shape: Vec<usize>| {
        expected_dense_with_gguf_shape("", name, shape.clone(), shape)
    };
    let matrix = |name: String, shape: Vec<usize>| ExpectedTensor {
        safetensors_name: String::new(),
        gguf_name: name,
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Matrix,
    };
    let hidden = vision.hidden_size as usize;
    let intermediate = vision.intermediate_size as usize;
    let patch = vision.patch_size as usize;
    let merged = hidden * (vision.merge_size as usize).pow(2);
    let mut expected = vec![
        dense("v.patch_embd.weight".into(), vec![hidden, 3, patch, patch]),
        dense("v.position_embd.weight".into(), vec![1024, hidden]),
        dense("v.pre_ln.weight".into(), vec![hidden]),
        dense("v.pre_ln.bias".into(), vec![hidden]),
        dense("v.post_ln.weight".into(), vec![hidden]),
        dense("v.post_ln.bias".into(), vec![hidden]),
    ];
    for layer in 0..vision.layer_count() {
        let prefix = format!("v.blk.{layer}");
        expected.extend([
            dense(format!("{prefix}.ln1.weight"), vec![hidden]),
            dense(format!("{prefix}.ln1.bias"), vec![hidden]),
            dense(format!("{prefix}.ln2.weight"), vec![hidden]),
            dense(format!("{prefix}.ln2.bias"), vec![hidden]),
            matrix(format!("{prefix}.attn_q.weight"), vec![hidden, hidden]),
            dense(format!("{prefix}.attn_q.bias"), vec![hidden]),
            matrix(format!("{prefix}.attn_k.weight"), vec![hidden, hidden]),
            dense(format!("{prefix}.attn_k.bias"), vec![hidden]),
            matrix(format!("{prefix}.attn_v.weight"), vec![hidden, hidden]),
            dense(format!("{prefix}.attn_v.bias"), vec![hidden]),
            matrix(format!("{prefix}.attn_out.weight"), vec![hidden, hidden]),
            dense(format!("{prefix}.attn_out.bias"), vec![hidden]),
            matrix(
                format!("{prefix}.ffn_up.weight"),
                vec![intermediate, hidden],
            ),
            dense(format!("{prefix}.ffn_up.bias"), vec![intermediate]),
            matrix(
                format!("{prefix}.ffn_down.weight"),
                vec![hidden, intermediate],
            ),
            dense(format!("{prefix}.ffn_down.bias"), vec![hidden]),
        ]);
    }
    expected.extend([
        matrix("mm.0.weight".into(), vec![4096, merged]),
        matrix("mm.1.weight".into(), vec![4096, 4096]),
        matrix("mm.2.weight".into(), vec![text.hidden_size as usize, 4096]),
    ]);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Muse-Glimmer projector");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Muse-Glimmer projector GGUF"));
        }
    }
    finish(issues)
}

fn validate_qwen3_vl_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = architecture.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    let is_moe = architecture == GgufArchitecture::Qwen3VlMoe;
    let metadata_name = architecture.metadata_name();
    let translate = |name: &str| dense_qwen::translate_gguf_weight_name(name, is_moe);
    if let Err(error) = checkpoint.catalog().translated_outputs(translate) {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args =
        match dense_qwen::config_from_gguf_catalog(checkpoint, metadata, metadata_name, is_moe) {
            Ok(args) => args,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    if let Err(error) = qwen3_vl::validate_qwen3_vl_text_gguf_catalog(&args, metadata) {
        return invalid_geometry(error.to_string());
    }
    let expected = dense_qwen_expected(&args);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Qwen3-VL text");
    if is_moe {
        issues.extend(validate_paired_expert_encodings(
            checkpoint,
            0..args.num_hidden_layers as usize,
            "Qwen3-VL-MoE",
        ));
    }
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Qwen3-VL text GGUF"));
        }
    }
    finish(issues)
}

pub(crate) fn validate_qwen3_vl_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let architecture = match dense_qwen::gguf_string(model_metadata, "general.architecture") {
        Ok(architecture) => architecture,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let is_moe = architecture == "qwen3vlmoe";
    if architecture != "qwen3vl" && !is_moe {
        return invalid_geometry(format!(
            "Qwen3-VL projector requires qwen3vl or qwen3vlmoe text, got {architecture:?}"
        ));
    }
    let text_args = match dense_qwen::config_from_gguf_catalog(
        model_checkpoint,
        model_metadata,
        &architecture,
        is_moe,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let args = match qwen3_vl::qwen3_vl_args_from_gguf_catalog(
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
        .translated_outputs(|name| qwen3_vl::translate_qwen3_vl_mmproj_name(name, &deepstack))
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let expected = qwen_vision_gguf_expected(&args.vision_config, args.text_config.hidden_size);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Qwen3-VL projector");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Qwen3-VL projector GGUF"));
        }
    }
    finish(issues)
}

pub(crate) fn validate_qwen35_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    let architecture = match model_metadata.get("general.architecture") {
        Some(GgufMetadataValue::String(architecture)) => architecture.clone(),
        Some(_) => {
            return invalid_geometry(
                "GGUF metadata key \"general.architecture\" has the wrong type".into(),
            )
        }
        None => {
            return invalid_geometry(
                "GGUF metadata is missing required key \"general.architecture\"".into(),
            )
        }
    };
    if !matches!(architecture.as_str(), "qwen35" | "qwen35moe") {
        return invalid_geometry(format!(
            "Qwen3.5 projector requires qwen35 or qwen35moe text, got {architecture:?}"
        ));
    }
    let text = match qwen35::qwen35_args_from_gguf_catalog(
        model_checkpoint,
        model_metadata,
        &architecture,
    ) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    let vision =
        match qwen3_vl::qwen_vision_config_from_gguf_catalog(checkpoint, metadata, "Qwen3.5") {
            Ok(vision) => vision,
            Err(error) => return invalid_geometry(error.to_string()),
        };
    if vision.out_hidden_size != text.hidden_size {
        return invalid_geometry(format!(
            "Qwen3.5 projector output {} does not match language hidden size {}",
            vision.out_hidden_size, text.hidden_size
        ));
    }
    if vision.deepstack_layer_count() != 0 {
        return invalid_geometry(format!(
            "Qwen3.5 projector declares {} DeepStack outputs; the decoder accepts only the primary merger output",
            vision.deepstack_layer_count()
        ));
    }
    let deepstack = vision.deepstack_layers();
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(|name| qwen3_vl::translate_qwen3_vl_mmproj_name(name, &deepstack))
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let expected = qwen_vision_gguf_expected(&vision, text.hidden_size);
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, "Qwen3.5 projector");
    for tensor in checkpoint.catalog().tensors() {
        let name = &tensor.descriptor().name;
        if !allowed.contains(name) {
            issues.push(unexpected_layout(name, "Qwen3.5 projector GGUF"));
        }
    }
    finish(issues)
}

fn qwen_vision_gguf_expected(
    vision: &qwen3_vl::VisionConfig,
    text_hidden_size: i32,
) -> Vec<ExpectedTensor> {
    let hidden = vision.hidden_size as usize;
    let intermediate = vision.intermediate_size as usize;
    let text_hidden = text_hidden_size as usize;
    let patch = vision.patch_size as usize;
    let merger_hidden = hidden * (vision.spatial_merge_size as usize).pow(2);
    let dense = |name: String, shape: Vec<usize>| {
        expected_dense_with_gguf_shape("", name, shape.clone(), shape)
    };
    let matrix = |name: String, shape: Vec<usize>| ExpectedTensor {
        safetensors_name: String::new(),
        gguf_name: name,
        safetensors_shape: shape.clone(),
        gguf_shape: shape,
        operation: TensorOperation::Matrix,
    };
    let mut tensors = vec![
        dense(
            "v.position_embd.weight".into(),
            vec![vision.num_position_embeddings as usize, hidden],
        ),
        dense("v.patch_embd.weight".into(), vec![hidden, 3, patch, patch]),
        dense(
            "v.patch_embd.weight.1".into(),
            vec![hidden, 3, patch, patch],
        ),
        dense("v.patch_embd.bias".into(), vec![hidden]),
    ];
    for layer in 0..vision.layer_count() {
        let prefix = format!("v.blk.{layer}");
        tensors.extend([
            dense(format!("{prefix}.ln1.weight"), vec![hidden]),
            dense(format!("{prefix}.ln1.bias"), vec![hidden]),
            matrix(
                format!("{prefix}.attn_qkv.weight"),
                vec![3 * hidden, hidden],
            ),
            dense(format!("{prefix}.attn_qkv.bias"), vec![3 * hidden]),
            matrix(format!("{prefix}.attn_out.weight"), vec![hidden, hidden]),
            dense(format!("{prefix}.attn_out.bias"), vec![hidden]),
            dense(format!("{prefix}.ln2.weight"), vec![hidden]),
            dense(format!("{prefix}.ln2.bias"), vec![hidden]),
            matrix(
                format!("{prefix}.ffn_up.weight"),
                vec![intermediate, hidden],
            ),
            dense(format!("{prefix}.ffn_up.bias"), vec![intermediate]),
            matrix(
                format!("{prefix}.ffn_down.weight"),
                vec![hidden, intermediate],
            ),
            dense(format!("{prefix}.ffn_down.bias"), vec![hidden]),
        ]);
    }
    tensors.extend([
        dense("v.post_ln.weight".into(), vec![hidden]),
        dense("v.post_ln.bias".into(), vec![hidden]),
        matrix("mm.0.weight".into(), vec![merger_hidden, merger_hidden]),
        dense("mm.0.bias".into(), vec![merger_hidden]),
        matrix("mm.2.weight".into(), vec![text_hidden, merger_hidden]),
        dense("mm.2.bias".into(), vec![text_hidden]),
    ]);
    for layer in vision.deepstack_layers() {
        let prefix = format!("v.deepstack.{layer}");
        tensors.extend([
            dense(format!("{prefix}.norm.weight"), vec![merger_hidden]),
            dense(format!("{prefix}.norm.bias"), vec![merger_hidden]),
            matrix(
                format!("{prefix}.fc1.weight"),
                vec![merger_hidden, merger_hidden],
            ),
            dense(format!("{prefix}.fc1.bias"), vec![merger_hidden]),
            matrix(
                format!("{prefix}.fc2.weight"),
                vec![text_hidden, merger_hidden],
            ),
            dense(format!("{prefix}.fc2.bias"), vec![text_hidden]),
        ]);
    }
    tensors
}

fn qwen35_gguf_expected(args: &qwen35::ModelArgs) -> Vec<ExpectedTensor> {
    let hidden = args.hidden_size as usize;
    let vocab = args.vocab_size as usize;
    let query = (args.num_attention_heads * args.head_dim) as usize;
    let key_value = (args.num_key_value_heads * args.head_dim) as usize;
    let key_dim = (args.linear_num_key_heads * args.linear_key_head_dim) as usize;
    let value_dim = (args.linear_num_value_heads * args.linear_value_head_dim) as usize;
    let value_heads = args.linear_num_value_heads as usize;
    let mut tensors = vec![
        expected(
            "model.embed_tokens.weight",
            "token_embd.weight",
            [vocab, hidden],
        ),
        expected_vector("model.norm.weight", "output_norm.weight", hidden),
    ];
    if !args.tie_word_embeddings {
        tensors.push(expected("lm_head.weight", "output.weight", [vocab, hidden]));
    }
    for layer in 0..args.num_hidden_layers as usize {
        let model = format!("model.layers.{layer}");
        let gguf = format!("blk.{layer}");
        tensors.extend([
            expected_vector(
                format!("{model}.input_layernorm.weight"),
                format!("{gguf}.attn_norm.weight"),
                hidden,
            ),
            expected_vector(
                format!("{model}.post_attention_layernorm.weight"),
                format!("{gguf}.post_attention_norm.weight"),
                hidden,
            ),
        ]);
        match args
            .layer_schedule
            .get(layer)
            .expect("validated Qwen hybrid layer schedule")
        {
            qwen35::LayerPolicy::SelfAttention(AttentionPolicy::Full) => {
                tensors.extend([
                    expected(
                        format!("{model}.self_attn.q_proj.weight"),
                        format!("{gguf}.attn_q.weight"),
                        [2 * query, hidden],
                    ),
                    expected(
                        format!("{model}.self_attn.k_proj.weight"),
                        format!("{gguf}.attn_k.weight"),
                        [key_value, hidden],
                    ),
                    expected(
                        format!("{model}.self_attn.v_proj.weight"),
                        format!("{gguf}.attn_v.weight"),
                        [key_value, hidden],
                    ),
                    expected(
                        format!("{model}.self_attn.o_proj.weight"),
                        format!("{gguf}.attn_output.weight"),
                        [hidden, query],
                    ),
                    expected_vector(
                        format!("{model}.self_attn.q_norm.weight"),
                        format!("{gguf}.attn_q_norm.weight"),
                        args.head_dim as usize,
                    ),
                    expected_vector(
                        format!("{model}.self_attn.k_norm.weight"),
                        format!("{gguf}.attn_k_norm.weight"),
                        args.head_dim as usize,
                    ),
                ]);
                if args.attention_bias {
                    tensors.extend([
                        expected_vector(
                            format!("{model}.self_attn.q_proj.bias"),
                            format!("{gguf}.attn_q.bias"),
                            2 * query,
                        ),
                        expected_vector(
                            format!("{model}.self_attn.k_proj.bias"),
                            format!("{gguf}.attn_k.bias"),
                            key_value,
                        ),
                        expected_vector(
                            format!("{model}.self_attn.v_proj.bias"),
                            format!("{gguf}.attn_v.bias"),
                            key_value,
                        ),
                        expected_vector(
                            format!("{model}.self_attn.o_proj.bias"),
                            format!("{gguf}.attn_output.bias"),
                            hidden,
                        ),
                    ]);
                }
            }
            qwen35::LayerPolicy::LinearAttention => {
                if args.model_type == "qwen3_next" {
                    tensors.extend([
                        expected(
                            format!("{model}.linear_attn.in_proj_qkvz.weight"),
                            format!("{gguf}.attn_qkvz.weight"),
                            [2 * key_dim + 2 * value_dim, hidden],
                        ),
                        expected(
                            format!("{model}.linear_attn.in_proj_ba.weight"),
                            format!("{gguf}.ssm_ba.weight"),
                            [2 * value_heads, hidden],
                        ),
                    ]);
                } else {
                    tensors.extend([
                        expected(
                            format!("{model}.linear_attn.in_proj_qkv.weight"),
                            format!("{gguf}.attn_qkv.weight"),
                            [2 * key_dim + value_dim, hidden],
                        ),
                        expected(
                            format!("{model}.linear_attn.in_proj_z.weight"),
                            format!("{gguf}.attn_gate.weight"),
                            [value_dim, hidden],
                        ),
                        expected(
                            format!("{model}.linear_attn.in_proj_b.weight"),
                            format!("{gguf}.ssm_beta.weight"),
                            [value_heads, hidden],
                        ),
                        expected(
                            format!("{model}.linear_attn.in_proj_a.weight"),
                            format!("{gguf}.ssm_alpha.weight"),
                            [value_heads, hidden],
                        ),
                    ]);
                }
                tensors.extend([
                    expected_dense_with_gguf_shape(
                        format!("{model}.linear_attn.conv1d.weight"),
                        format!("{gguf}.ssm_conv1d.weight"),
                        vec![
                            2 * key_dim + value_dim,
                            1,
                            args.linear_conv_kernel_dim as usize,
                        ],
                        vec![
                            2 * key_dim + value_dim,
                            args.linear_conv_kernel_dim as usize,
                        ],
                    ),
                    expected_vector_shape(format!("{gguf}.ssm_dt.bias"), vec![value_heads]),
                    ExpectedTensor {
                        safetensors_name: format!("{model}.linear_attn.A_log"),
                        gguf_name: format!("{gguf}.ssm_a"),
                        safetensors_shape: vec![value_heads],
                        gguf_shape: vec![value_heads],
                        operation: TensorOperation::Dense,
                    },
                    expected_vector(
                        format!("{model}.linear_attn.norm.weight"),
                        format!("{gguf}.ssm_norm.weight"),
                        args.linear_value_head_dim as usize,
                    ),
                    expected(
                        format!("{model}.linear_attn.out_proj.weight"),
                        format!("{gguf}.ssm_out.weight"),
                        [hidden, value_dim],
                    ),
                ]);
            }
            qwen35::LayerPolicy::SelfAttention(AttentionPolicy::Sliding { .. }) => {
                unreachable!("Qwen hybrid validation rejects sliding self-attention")
            }
        }
        if args.is_moe() {
            let experts = args.num_experts as usize;
            let intermediate = args.moe_intermediate_size as usize;
            let shared = args.shared_expert_intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{model}.mlp.gate.weight"),
                    format!("{gguf}.ffn_gate_inp.weight"),
                    [experts, hidden],
                ),
                expected(
                    format!("{model}.mlp.shared_expert.gate_proj.weight"),
                    format!("{gguf}.ffn_gate_shexp.weight"),
                    [shared, hidden],
                ),
                expected(
                    format!("{model}.mlp.shared_expert.up_proj.weight"),
                    format!("{gguf}.ffn_up_shexp.weight"),
                    [shared, hidden],
                ),
                expected(
                    format!("{model}.mlp.shared_expert.down_proj.weight"),
                    format!("{gguf}.ffn_down_shexp.weight"),
                    [hidden, shared],
                ),
                expected(
                    format!("{model}.mlp.shared_expert_gate.weight"),
                    format!("{gguf}.ffn_gate_inp_shexp.weight"),
                    [1, hidden],
                ),
                expected_rank3(
                    format!("{model}.mlp.experts.gate_proj"),
                    format!("{gguf}.ffn_gate_exps.weight"),
                    [experts, intermediate, hidden],
                ),
                expected_rank3(
                    format!("{model}.mlp.experts.up_proj"),
                    format!("{gguf}.ffn_up_exps.weight"),
                    [experts, intermediate, hidden],
                ),
                expected_rank3(
                    format!("{model}.mlp.experts.down_proj"),
                    format!("{gguf}.ffn_down_exps.weight"),
                    [experts, hidden, intermediate],
                ),
            ]);
        } else {
            let intermediate = args.intermediate_size as usize;
            tensors.extend([
                expected(
                    format!("{model}.mlp.gate_proj.weight"),
                    format!("{gguf}.ffn_gate.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{model}.mlp.up_proj.weight"),
                    format!("{gguf}.ffn_up.weight"),
                    [intermediate, hidden],
                ),
                expected(
                    format!("{model}.mlp.down_proj.weight"),
                    format!("{gguf}.ffn_down.weight"),
                    [hidden, intermediate],
                ),
            ]);
        }
    }
    tensors
}

fn validate_qwen35_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let metadata_name = architecture.metadata_name();
    let is_next = architecture == GgufArchitecture::Qwen3Next;
    let loader_name = if is_next { "Qwen3-Next" } else { "Qwen3.5" };
    if let Err(error) = checkpoint
        .catalog()
        .translated_outputs(qwen35::qwen35_translate_gguf_weight_name)
    {
        return StructuralValidation::Invalid(vec![StructuralIssue {
            kind: StructuralIssueKind::ConflictingLayout,
            detail: error.to_string(),
            tensor_name: None,
            tensor_type_code: None,
            metadata_key: None,
        }]);
    }
    let args = match qwen35::qwen35_args_from_gguf_catalog(checkpoint, metadata, metadata_name) {
        Ok(args) => args,
        Err(error) => return invalid_geometry(error.to_string()),
    };
    if !args.is_moe() && options.weight_residency.expert_cache().is_some() {
        return invalid_geometry(format!(
            "sparse expert caching requires a {loader_name} MoE GGUF checkpoint"
        ));
    }
    let mut expected = qwen35_gguf_expected(&args);
    let actual_shapes = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| {
            (
                tensor.descriptor().name.as_str(),
                tensor.descriptor().mlx_shape(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    for tensor in &mut expected {
        if tensor.gguf_name.ends_with("ffn_gate_inp_shexp.weight")
            && actual_shapes
                .get(tensor.gguf_name.as_str())
                .is_some_and(|shape| shape.as_slice() == [args.hidden_size as u64])
        {
            tensor.gguf_shape = vec![args.hidden_size as usize];
        }
    }
    let allowed = expected
        .iter()
        .map(|tensor| tensor.gguf_name.clone())
        .collect::<BTreeSet<_>>();
    let mut issues = validate_gguf_plan(checkpoint, expected, loader_name);
    for actual in checkpoint.catalog().tensors() {
        let name = &actual.descriptor().name;
        let nextn_or_unused_block = name
            .strip_prefix("blk.")
            .and_then(|rest| rest.split('.').next())
            .and_then(|index| index.parse::<usize>().ok())
            .is_some_and(|index| index >= args.num_hidden_layers as usize);
        if allowed.contains(name) || name.starts_with("rope_freqs.") || nextn_or_unused_block {
            continue;
        }
        issues.push(unexpected_layout(name, &format!("{loader_name} GGUF")));
    }
    if args.is_moe() {
        issues.extend(validate_paired_expert_encodings(
            checkpoint,
            0..args.num_hidden_layers as usize,
            &format!("{loader_name} MoE"),
        ));
    }
    for layer in 0..args.num_hidden_layers as usize {
        if args.layer_schedule.get(layer) != Some(&qwen35::LayerPolicy::LinearAttention) {
            continue;
        }
        if is_next {
            for name in [
                format!("blk.{layer}.attn_qkvz.weight"),
                format!("blk.{layer}.ssm_ba.weight"),
            ] {
                let Some(tensor) = checkpoint
                    .catalog()
                    .tensors()
                    .find(|tensor| tensor.descriptor().name == name)
                else {
                    continue;
                };
                let Some((_, group_size)) = tensor.affine() else {
                    continue;
                };
                if !(args.hidden_size as u32).is_multiple_of(group_size) {
                    issues.push(StructuralIssue {
                        kind: StructuralIssueKind::QuantizationCompanionMismatch,
                        detail: format!(
                            "Qwen3-Next GGUF fused projection {name:?} input dimension {} is not divisible by group size {group_size}",
                            args.hidden_size
                        ),
                        tensor_name: Some(name),
                        tensor_type_code: Some(tensor.descriptor().ggml_type.code()),
                        metadata_key: None,
                    });
                }
            }
            continue;
        }
        let name = format!("blk.{layer}.ssm_out.weight");
        let Some(tensor) = checkpoint
            .catalog()
            .tensors()
            .find(|tensor| tensor.descriptor().name == name)
        else {
            continue;
        };
        let Some((bits, group_size)) = tensor.affine() else {
            continue;
        };
        let head = args.linear_value_head_dim as u32;
        if head
            .checked_mul(u32::from(bits))
            .is_none_or(|width| !width.is_multiple_of(32))
            || !head.is_multiple_of(group_size)
        {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::QuantizationCompanionMismatch,
                detail: format!(
                    "Qwen3.5 GGUF tensor {name:?} cannot preserve grouped value-head layout with {bits}-bit groups of {group_size}"
                ),
                tensor_name: Some(name),
                tensor_type_code: Some(tensor.descriptor().ggml_type.code()),
                metadata_key: None,
            });
        }
    }
    finish(issues)
}

fn validate_paired_expert_encodings(
    checkpoint: &GgufCheckpoint,
    layers: impl IntoIterator<Item = usize>,
    loader_name: &str,
) -> Vec<StructuralIssue> {
    let catalog = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| (tensor.descriptor().name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let mut issues = Vec::new();
    for layer in layers {
        let gate_name = format!("blk.{layer}.ffn_gate_exps.weight");
        let up_name = format!("blk.{layer}.ffn_up_exps.weight");
        let (Some(gate), Some(up)) = (
            catalog.get(gate_name.as_str()),
            catalog.get(up_name.as_str()),
        ) else {
            continue;
        };
        if gate.descriptor().ggml_type != up.descriptor().ggml_type
            || gate.affine() != up.affine()
            || gate.is_mxfp4() != up.is_mxfp4()
        {
            issues.push(StructuralIssue {
                kind: StructuralIssueKind::QuantizationCompanionMismatch,
                detail: format!(
                    "{loader_name} paired expert tensors {gate_name:?} and {up_name:?} use incompatible encodings {:?} and {:?}",
                    gate.descriptor().ggml_type,
                    up.descriptor().ggml_type
                ),
                tensor_name: Some(gate_name),
                tensor_type_code: Some(gate.descriptor().ggml_type.code()),
                metadata_key: None,
            });
        }
    }
    issues
}

fn validate_gguf_plan(
    checkpoint: &GgufCheckpoint,
    expected: Vec<ExpectedTensor>,
    loader_name: &str,
) -> Vec<StructuralIssue> {
    let constraints = expected
        .into_iter()
        .map(|expected| {
            GgufTensorConstraint::required(
                expected.gguf_name,
                expected.gguf_shape,
                GgufTypeConstraint::OperationClass(expected.operation),
            )
        })
        .collect();
    let plan = GgufCheckpointPlan::new(
        loader_name,
        constraints,
        Vec::new(),
        CatalogPolicy::non_strict(),
    )
    .expect("legacy GGUF structural constraints are valid");
    match checkpoint_validation::validate_gguf_plan(checkpoint, &plan) {
        StructuralValidation::Exact => Vec::new(),
        StructuralValidation::Invalid(issues) => issues,
        StructuralValidation::Unverified(issue) => vec![issue],
    }
}

fn validate_gguf_one_of(
    checkpoint: &GgufCheckpoint,
    name: &str,
    expected_shapes: &[Vec<usize>],
    operation: TensorOperation,
    loader_name: &str,
) -> Vec<StructuralIssue> {
    let Some(actual) = checkpoint
        .catalog()
        .tensors()
        .find(|tensor| tensor.descriptor().name == name)
    else {
        return vec![missing(name)];
    };
    let shape = actual
        .descriptor()
        .mlx_shape()
        .into_iter()
        .map(|dimension| usize::try_from(dimension).unwrap_or(usize::MAX))
        .collect::<Vec<_>>();
    let mut issues = Vec::new();
    if !expected_shapes.contains(&shape) {
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::ShapeMismatch,
            detail: format!(
                "tensor {name:?} expected one of shapes {expected_shapes:?}, got {shape:?}"
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: None,
            metadata_key: None,
        });
    }
    if !gguf_encoding_supported(operation, actual.descriptor().ggml_type) {
        let encoding = actual.descriptor().ggml_type;
        issues.push(StructuralIssue {
            kind: StructuralIssueKind::UnsupportedEncoding,
            detail: format!(
                "GGUF tensor {name:?} uses {encoding:?} (type {}) for a {operation:?} operation, which the {loader_name} loader does not support",
                encoding.code()
            ),
            tensor_name: Some(name.into()),
            tensor_type_code: Some(encoding.code()),
            metadata_key: None,
        });
    }
    issues
}

fn validate_inkling_paired_gguf_formats(
    checkpoint: &GgufCheckpoint,
    gate_name: &str,
    up_name: &str,
) -> Vec<StructuralIssue> {
    let catalog = checkpoint
        .catalog()
        .tensors()
        .map(|tensor| (tensor.descriptor().name.as_str(), tensor))
        .collect::<BTreeMap<_, _>>();
    let (Some(gate), Some(up)) = (catalog.get(gate_name), catalog.get(up_name)) else {
        return Vec::new();
    };
    let dense = |encoding| matches!(encoding, GgufType::F32 | GgufType::F16 | GgufType::Bf16);
    let gate_type = gate.descriptor().ggml_type;
    let up_type = up.descriptor().ggml_type;
    let compatible = if dense(gate_type) && dense(up_type) {
        true
    } else {
        gate_type == up_type && gate.affine() == up.affine() && gate.is_mxfp4() == up.is_mxfp4()
    };
    if compatible {
        Vec::new()
    } else {
        vec![StructuralIssue {
            kind: StructuralIssueKind::QuantizationCompanionMismatch,
            detail: format!(
                "Inkling paired expert tensors {gate_name:?} and {up_name:?} use incompatible encodings {gate_type:?} and {up_type:?}"
            ),
            tensor_name: Some(gate_name.into()),
            tensor_type_code: Some(gate_type.code()),
            metadata_key: None,
        }]
    }
}

fn is_float_dtype(dtype: &StoredDtype) -> bool {
    matches!(
        dtype,
        StoredDtype::F16 | StoredDtype::BF16 | StoredDtype::F32
    )
}

fn finish(issues: Vec<StructuralIssue>) -> StructuralValidation {
    if issues.is_empty() {
        StructuralValidation::Exact
    } else {
        StructuralValidation::Invalid(issues)
    }
}

fn invalid_geometry(detail: String) -> StructuralValidation {
    StructuralValidation::Invalid(vec![StructuralIssue {
        kind: StructuralIssueKind::InvalidGeometry,
        detail,
        tensor_name: None,
        tensor_type_code: None,
        metadata_key: None,
    }])
}

fn missing(name: &str) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::MissingTensor,
        detail: format!("checkpoint is missing required tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn layout(name: &str, detail: String) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::ConflictingLayout,
        detail: format!("could not validate tensor {name:?}: {detail}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn unexpected_layout(name: &str, loader_name: &str) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::UnexpectedTensor,
        detail: format!("{loader_name} catalog contains unexpected tensor {name:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

fn shape_mismatch(name: &str, expected: &[usize], actual: &[usize]) -> StructuralIssue {
    StructuralIssue {
        kind: StructuralIssueKind::ShapeMismatch,
        detail: format!("tensor {name:?} expected shape {expected:?}, got {actual:?}"),
        tensor_name: Some(name.into()),
        tensor_type_code: None,
        metadata_key: None,
    }
}

#[cfg(test)]
mod admission_policy_tests {
    use super::*;

    #[test]
    fn non_strict_catalog_ignores_only_unexpected_tensors() {
        let unexpected = unexpected_layout("unrelated.weight", "test");
        let malformed = shape_mismatch("model.weight", &[2, 2], &[1]);
        assert_eq!(
            StructuralValidation::Invalid(vec![unexpected.clone(), malformed.clone()])
                .with_strict_catalog(false),
            StructuralValidation::Invalid(vec![malformed])
        );

        let error = StructuralValidation::Invalid(vec![unexpected])
            .into_loader_result()
            .unwrap_err();
        assert!(matches!(
            error,
            Error::StrictLoadValidation { missing, unused }
                if missing.is_empty() && unused == ["unrelated.weight"]
        ));
    }
}

#[cfg(test)]
mod dense_qwen_tests {
    use super::*;

    fn qwen2_args(tied: bool) -> dense_qwen::DecoderConfig {
        dense_qwen::config_from_hf_value(&serde_json::json!({
            "model_type": "qwen2", "hidden_size": 8, "num_hidden_layers": 2,
            "intermediate_size": 16, "num_attention_heads": 4,
            "num_key_value_heads": 2, "rms_norm_eps": 1e-6, "vocab_size": 32,
            "max_position_embeddings": 64, "rope_theta": 10000.0,
            "tie_word_embeddings": tied, "use_sliding_window": false
        }))
        .unwrap()
    }

    #[test]
    fn qwen2_plan_is_exactly_biased_and_has_no_qk_norms() {
        let tied = dense_qwen_expected(&qwen2_args(true));
        let names = tied
            .iter()
            .map(|tensor| tensor.safetensors_name.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.self_attn.q_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.k_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.v_proj.bias"));
        assert!(!names.contains("model.layers.0.self_attn.q_norm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.k_norm.weight"));
        assert!(!names.contains("lm_head.weight"));

        let untied = dense_qwen_expected(&qwen2_args(false));
        assert!(untied
            .iter()
            .any(|tensor| tensor.safetensors_name == "lm_head.weight"));
    }
}

#[cfg(test)]
mod lfm2_schedule_tests {
    use super::*;

    #[test]
    fn structural_plan_uses_each_feed_forward_policy_in_order() {
        let mut args = lfm2::model_args_from_config_value(&serde_json::json!({
            "model_type": "lfm2_moe", "vocab_size": 32, "hidden_size": 16,
            "intermediate_size": 24, "num_hidden_layers": 3,
            "num_attention_heads": 4, "num_key_value_heads": 2,
            "max_position_embeddings": 64, "norm_eps": 1e-5,
            "layer_types": ["conv", "full_attention", "conv"],
            "conv_L_cache": 3, "block_auto_adjust_ff_dim": false,
            "moe_intermediate_size": 8, "num_dense_layers": 1,
            "num_experts": 2, "num_experts_per_tok": 1
        }))
        .unwrap();
        args.layer_schedule = crate::LayerSchedule::new(
            3,
            vec![
                lfm2::LayerPolicy {
                    operator: lfm2::OperatorPolicy::CausalConvolution,
                    feed_forward: lfm2::FeedForwardPolicy::SparseMoe,
                },
                lfm2::LayerPolicy {
                    operator: lfm2::OperatorPolicy::SelfAttention(crate::AttentionPolicy::Full),
                    feed_forward: lfm2::FeedForwardPolicy::Dense,
                },
                lfm2::LayerPolicy {
                    operator: lfm2::OperatorPolicy::CausalConvolution,
                    feed_forward: lfm2::FeedForwardPolicy::SparseMoe,
                },
            ],
        )
        .unwrap();

        let names = lfm2_expected(&args)
            .unwrap()
            .into_iter()
            .map(|tensor| tensor.safetensors_name)
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.feed_forward.gate.weight"));
        assert!(!names.contains("model.layers.0.feed_forward.w1.weight"));
        assert!(names.contains("model.layers.1.feed_forward.w1.weight"));
        assert!(!names.contains("model.layers.1.feed_forward.gate.weight"));
        assert!(names.contains("model.layers.2.feed_forward.gate.weight"));
    }
}
