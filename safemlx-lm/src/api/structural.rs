//! Pure checkpoint-structure plans shared by inspection and high-level loading.

use std::collections::{BTreeSet, HashMap};
use std::path::Path;

use safemlx::ops::{GgufCheckpoint, GgufMetadataValue};
use serde_json::Value;

use super::{GgufArchitecture, ModelKind, ModelLoadOptions};
use crate::{
    architectures::{
        deepseek_v3::checkpoint as deepseek_v3_checkpoint,
        deepseek_v4::checkpoint as deepseek_v4_checkpoint,
        gemma4::{checkpoint as gemma4_checkpoint, model as gemma4},
        gpt_oss::checkpoint as gpt_oss_checkpoint,
        inkling::{checkpoint as inkling_checkpoint, model as inkling},
        kimi_linear::checkpoint as kimi_linear_checkpoint,
        lfm2::model as lfm2,
        llama::model as llama,
        moshi::personaplex,
        muse_glimmer,
        nemotron_h::model as nemotron_h,
        qwen::{
            dense::checkpoint as dense_qwen_checkpoint,
            hybrid::checkpoint as qwen_hybrid_checkpoint, vl::checkpoint as qwen_vl_checkpoint,
        },
    },
    error::Error,
    runtime::checkpoint::{
        schema::{
            AlternativeLayoutGroup, CatalogPolicy, GgufCheckpointPlan, GgufTensorConstraint,
            GgufTypeConstraint, LayoutVariant, SafetensorsCheckpointPlan,
            SafetensorsTensorConstraint, StoredDtypeConstraint, TensorOperation,
        },
        store::{SafetensorsWeightStore, StoredDtype, WeightStore},
        validation as checkpoint_validation,
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
            ModelKind::KimiLinear => kimi_linear_checkpoint::validate_safetensors(config, store),
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
            GgufArchitecture::KimiLinear => {
                kimi_linear_checkpoint::validate_gguf(checkpoint, metadata)
            }
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
    inkling_checkpoint::validate_safetensors(config, store)
}

fn validate_gemma4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    gemma4_checkpoint::validate_safetensors(
        config,
        store,
        !options.weight_residency.is_fully_resident(),
        options.weight_residency.expert_cache().is_some(),
    )
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

fn validate_deepseek_v4_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
) -> StructuralValidation {
    deepseek_v4_checkpoint::validate_safetensors(config, store)
}

fn validate_deepseek_v3_safetensors(
    config: &Value,
    store: &SafetensorsWeightStore,
    options: ModelLoadOptions,
) -> StructuralValidation {
    deepseek_v3_checkpoint::validate_safetensors(
        config,
        store,
        !options.weight_residency.is_fully_resident(),
    )
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
    gpt_oss_checkpoint::validate_safetensors(config, store)
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
    dense_qwen_checkpoint::validate_safetensors(config, store)
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
    qwen_vl_checkpoint::validate_safetensors(
        kind == ModelKind::Qwen3VlMoe,
        config,
        store,
        !options.weight_residency.is_fully_resident(),
    )
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
    deepseek_v3_checkpoint::validate_gguf(checkpoint, metadata)
}

fn validate_deepseek4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    deepseek_v4_checkpoint::validate_gguf(checkpoint, metadata)
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
        issues.extend(checkpoint_validation::validate_matching_gguf_encodings(
            checkpoint,
            args.layer_schedule
                .iter()
                .enumerate()
                .filter_map(|(layer, policy)| {
                    (policy.feed_forward == lfm2::FeedForwardPolicy::SparseMoe).then_some((
                        format!("blk.{layer}.ffn_gate_exps.weight"),
                        format!("blk.{layer}.ffn_up_exps.weight"),
                    ))
                }),
            "LFM2 MoE",
        ));
    }
    finish(issues)
}

fn validate_inkling_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = GgufArchitecture::Inkling.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    inkling_checkpoint::validate_gguf(checkpoint, metadata)
}

pub(crate) fn validate_inkling_mmproj_gguf(
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &inkling::InklingMmprojGguf,
) -> StructuralValidation {
    inkling_checkpoint::validate_mmproj_gguf(model_metadata, mmproj)
}

pub(crate) fn validate_gemma4_mmproj_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    mmproj: &gemma4::Gemma4MmprojGguf,
) -> StructuralValidation {
    gemma4_checkpoint::validate_mmproj_gguf(model_checkpoint, model_metadata, mmproj)
}

fn validate_gemma4_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    if let Err(error) = GgufArchitecture::Gemma4.validate_load_policy(options) {
        return invalid_geometry(error.to_string());
    }
    gemma4_checkpoint::validate_gguf(checkpoint, metadata)
}

fn validate_gpt_oss_gguf(
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    gpt_oss_checkpoint::validate_gguf(checkpoint, metadata)
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
    let variant = match architecture {
        GgufArchitecture::Qwen2 => dense_qwen_checkpoint::GgufVariant::Qwen2,
        GgufArchitecture::Qwen3 => dense_qwen_checkpoint::GgufVariant::Qwen3,
        GgufArchitecture::Qwen3Moe => dense_qwen_checkpoint::GgufVariant::Qwen3Moe,
        _ => unreachable!("dense Qwen GGUF validator received another architecture"),
    };
    dense_qwen_checkpoint::validate_gguf(variant, checkpoint, metadata)
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
    let variant = match architecture {
        GgufArchitecture::Qwen3Vl => qwen_vl_checkpoint::GgufVariant::Dense,
        GgufArchitecture::Qwen3VlMoe => qwen_vl_checkpoint::GgufVariant::Moe,
        _ => unreachable!("Qwen-VL GGUF validator received another architecture"),
    };
    qwen_vl_checkpoint::validate_gguf(variant, checkpoint, metadata)
}

pub(crate) fn validate_qwen3_vl_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    qwen_vl_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}

pub(crate) fn validate_qwen35_projector_gguf(
    model_checkpoint: &GgufCheckpoint,
    model_metadata: &HashMap<String, GgufMetadataValue>,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
) -> StructuralValidation {
    qwen_hybrid_checkpoint::validate_projector_gguf(
        model_checkpoint,
        model_metadata,
        checkpoint,
        metadata,
    )
}
fn validate_qwen35_gguf(
    architecture: GgufArchitecture,
    checkpoint: &GgufCheckpoint,
    metadata: &HashMap<String, GgufMetadataValue>,
    options: ModelLoadOptions,
) -> StructuralValidation {
    let variant = match architecture {
        GgufArchitecture::Qwen35 => qwen_hybrid_checkpoint::GgufVariant::Qwen35,
        GgufArchitecture::Qwen35Moe => qwen_hybrid_checkpoint::GgufVariant::Qwen35Moe,
        GgufArchitecture::Qwen3Next => qwen_hybrid_checkpoint::GgufVariant::Qwen3Next,
        _ => unreachable!("Qwen hybrid GGUF validator received another architecture"),
    };
    qwen_hybrid_checkpoint::validate_gguf(
        variant,
        checkpoint,
        metadata,
        options.weight_residency.expert_cache().is_some(),
    )
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

#[cfg(test)]
mod admission_policy_tests {
    use super::*;

    #[test]
    fn non_strict_catalog_ignores_only_unexpected_tensors() {
        let unexpected = unexpected_layout("unrelated.weight", "test");
        let malformed =
            crate::runtime::checkpoint::contract::shape_mismatch("model.weight", &[2, 2], &[1]);
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
    use crate::architectures::qwen::dense;

    fn qwen2_args(tied: bool) -> dense::DecoderConfig {
        dense::config_from_hf_value(&serde_json::json!({
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
        let tied = dense_qwen_checkpoint::safetensors_plan(&qwen2_args(true)).unwrap();
        let names = tied
            .common_tensors
            .iter()
            .map(|tensor| tensor.key.as_str())
            .collect::<BTreeSet<_>>();
        assert!(names.contains("model.layers.0.self_attn.q_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.k_proj.bias"));
        assert!(names.contains("model.layers.0.self_attn.v_proj.bias"));
        assert!(!names.contains("model.layers.0.self_attn.q_norm.weight"));
        assert!(!names.contains("model.layers.0.self_attn.k_norm.weight"));
        assert!(!names.contains("lm_head.weight"));

        let untied = dense_qwen_checkpoint::safetensors_plan(&qwen2_args(false)).unwrap();
        assert!(untied
            .common_tensors
            .iter()
            .any(|tensor| tensor.key == "lm_head.weight"));
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
