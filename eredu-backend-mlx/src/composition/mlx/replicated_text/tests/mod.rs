use super::*;

mod admission_failures;
mod execution;
mod selection_and_lowering;
mod state_and_controls;

use crate::backend::ExecutionContext;
use eredu_checkpoint::SourceTensorEncoding;
use eredu_core::{
    cache::LayerCachePolicy, AttentionPolicy, LayerSchedule, ModelConfigurationResolver,
};
use eredu_runtime::{
    ParameterTransformConstraint, ReplicatedTextParameterRequirement, ReplicatedTextStateAccess,
    StateLayout, WeightLoweringDescriptor,
};
use safemlx::{Device, DeviceType};

fn prediction_cache_manager() -> CacheResidencyManager {
    CacheResidencyManager::new(
        PagedCacheOptions::new(1, 1 << 20, 1 << 20, 1)
            .unwrap()
            .with_full_attention(true),
    )
    .unwrap()
}

struct ForgedShapeSource {
    inner: eredu_checkpoint::store::SharedCheckpointSource,
    key: String,
}

impl eredu_checkpoint::store::CheckpointSource for ForgedShapeSource {
    fn source_keys(&self) -> Vec<String> {
        self.inner.source_keys()
    }

    fn source_metadata(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorMetadata, eredu_checkpoint::store::StoreError> {
        let mut metadata = self.inner.source_metadata(key)?;
        if key == self.key {
            metadata.physical_shape.push(2);
        }
        Ok(metadata)
    }

    fn acquire_lease(
        &self,
        request: eredu_checkpoint::store::TensorReadRequest,
    ) -> Result<eredu_checkpoint::store::CheckpointLease, eredu_checkpoint::store::StoreError> {
        self.inner.acquire_lease(request)
    }

    fn source_diagnostics(
        &self,
    ) -> Result<eredu_checkpoint::store::WeightStoreDiagnostics, eredu_checkpoint::store::StoreError>
    {
        self.inner.source_diagnostics()
    }

    fn source_provenance(
        &self,
        key: &str,
    ) -> Result<eredu_checkpoint::store::TensorSourceProvenance, eredu_checkpoint::store::StoreError>
    {
        self.inner.source_provenance(key)
    }
}

fn materialize_model_plan(
    plan: eredu_core::ModelPreparationPlan<
        eredu_architectures::processor_plan::ArtifactArchitecturePlan,
    >,
    options: crate::MlxLoadRequest,
    stream: &Stream,
    weights_stream: &Stream,
) -> Result<crate::backend::MlxModel, crate::backend::error::Error> {
    let selected = super::super::loading::select_preparation(plan.inspection(), options)?;
    let plan = eredu_core::plan_model_preparation(
        plan.inspection().clone(),
        plan.policy(),
        selected.session_capabilities(),
    )?;
    let (sources, _rank_context) = super::super::loading::prepare_selected_sources(plan, selected)?;
    super::super::loading::materialize_model_plan(sources, None, stream, weights_stream)
}

pub(crate) fn tiny_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
    tiny_safetensors_artifact(model_type, tied, false, false)
}

fn tiny_sharded_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
    tiny_safetensors_artifact(model_type, tied, true, false)
}

fn tiny_packed_safetensors_artifact(model_type: &str) -> tempfile::TempDir {
    tiny_safetensors_artifact(model_type, false, false, true)
}

fn tiny_safetensors_artifact(
    model_type: &str,
    tied: bool,
    sharded: bool,
    packed: bool,
) -> tempfile::TempDir {
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

    let root = tempfile::tempdir().unwrap();
    let architecture = match model_type {
        "llama" => "LlamaForCausalLM",
        "mistral" => "MistralForCausalLM",
        "qwen2" => "Qwen2ForCausalLM",
        "qwen3" => "Qwen3ForCausalLM",
        "qwen3_moe" => "Qwen3MoeForCausalLM",
        "gpt_oss" => "GptOssForCausalLM",
        _ => unreachable!(),
    };
    let mut config = serde_json::json!({
        "model_type": model_type,
        "architectures": [architecture],
        "hidden_size": 32,
        "num_hidden_layers": 1,
        "intermediate_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 1,
        "head_dim": 8,
        "rms_norm_eps": 0.00001,
        "vocab_size": 64,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "tie_word_embeddings": tied
    });
    if model_type == "qwen3_moe" {
        config["num_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        config["moe_intermediate_size"] = 32.into();
    }
    if model_type == "gpt_oss" {
        config["num_local_experts"] = 2.into();
        config["num_experts_per_tok"] = 1.into();
        config["sliding_window"] = 16.into();
        config["layer_types"] = serde_json::json!(["sliding_attention"]);
        config["quantization_config"] = serde_json::json!({"quant_method": "mxfp4"});
        config["swiglu_limit"] = 7.0.into();
    }
    if packed {
        config["quantization_config"] = serde_json::json!({ "group_size": 32, "bits": 4 });
    }
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let plan = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    constraints.extend(
        plan.layout_groups
            .iter()
            .filter(|group| group.required)
            .filter_map(|group| group.variants.first())
            .flat_map(|variant| variant.tensors.iter()),
    );
    let tensors = constraints
        .into_iter()
        .filter(|constraint| {
            constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
        })
        .map(|constraint| {
            let elements = constraint.shape.iter().product::<usize>();
            let (dtype, bytes): (Dtype, Vec<u8>) = if matches!(
                constraint.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U32
                )
            ) {
                (
                    Dtype::U32,
                    (0..elements).flat_map(|_| 0_u32.to_le_bytes()).collect(),
                )
            } else if matches!(
                constraint.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U8
                )
            ) {
                let fill = if constraint.key.ends_with("_scales") {
                    127
                } else {
                    0
                };
                (Dtype::U8, vec![fill; elements])
            } else if constraint.key.ends_with(".A_log") {
                (
                    Dtype::F32,
                    (-1.0_f32)
                        .to_le_bytes()
                        .into_iter()
                        .cycle()
                        .take(elements * 4)
                        .collect(),
                )
            } else if constraint.key.contains("norm") && constraint.key.ends_with(".weight") {
                (
                    Dtype::F32,
                    (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                )
            } else if constraint.role == eredu_checkpoint::schema::TensorRole::Companion {
                (
                    Dtype::F32,
                    (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                )
            } else {
                let seed = constraint.key.bytes().fold(1_u32, |value, byte| {
                    value.wrapping_mul(31) ^ u32::from(byte)
                });
                (
                    Dtype::F32,
                    (0..elements)
                        .flat_map(|index| {
                            let signed = i32::try_from((seed as usize + index) % 29).unwrap() - 14;
                            (signed as f32 * 0.001).to_le_bytes()
                        })
                        .collect(),
                )
            };
            (
                constraint.key.clone(),
                constraint.shape.clone(),
                dtype,
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let write = |path: &Path, tensors: &[(String, Vec<usize>, Dtype, Vec<u8>)]| {
        let views = tensors
            .iter()
            .map(|(name, shape, dtype, bytes)| {
                (
                    name.as_str(),
                    TensorView::new(*dtype, shape.clone(), bytes.as_slice()).unwrap(),
                )
            })
            .collect::<Vec<_>>();
        serialize_to_file(views, None, path).unwrap();
    };
    if sharded {
        let split = tensors.len() / 2;
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        write(&root.path().join(first), &tensors[..split]);
        write(&root.path().join(second), &tensors[split..]);
        let weight_map = tensors
            .iter()
            .enumerate()
            .map(|(index, (name, _, _, _))| {
                (name.clone(), if index < split { first } else { second })
            })
            .collect::<std::collections::BTreeMap<_, _>>();
        std::fs::write(
            root.path().join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({ "weight_map": weight_map })).unwrap(),
        )
        .unwrap();
    } else {
        write(&root.path().join("model.safetensors"), &tensors);
    }
    root
}

fn tiny_heterogeneous_artifact(config: serde_json::Value) -> tempfile::TempDir {
    tiny_heterogeneous_artifact_with_layout(config, false)
}

fn tiny_heterogeneous_artifact_with_layout(
    config: serde_json::Value,
    fused_qwen_next: bool,
) -> tempfile::TempDir {
    use safetensors::{tensor::serialize_to_file, tensor::TensorView, Dtype};

    let root = tempfile::tempdir().unwrap();
    std::fs::write(
        root.path().join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    let resolved = eredu_architectures::configuration::MODEL_CONFIGURATIONS
        .resolve_safetensors(&config)
        .unwrap();
    let plan = resolved
        .architecture_plan()
        .safetensors_architecture()
        .unwrap()
        .checkpoint();
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    constraints.extend(
        plan.layout_groups
            .iter()
            .filter(|group| group.required)
            .filter_map(|group| {
                if fused_qwen_next && group.variants.iter().any(|variant| variant.id == "fused") {
                    group.variants.iter().find(|variant| variant.id == "fused")
                } else {
                    group.variants.first()
                }
            })
            .flat_map(|variant| variant.tensors.iter()),
    );
    let tensors = constraints
        .into_iter()
        .filter(|constraint| {
            constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
        })
        .map(|constraint| {
            let elements = constraint.shape.iter().product::<usize>();
            let (dtype, bytes): (Dtype, Vec<u8>) = if matches!(
                constraint.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::U32
                )
            ) {
                (
                    Dtype::U32,
                    (0..elements).flat_map(|_| 0_u32.to_le_bytes()).collect(),
                )
            } else if matches!(
                constraint.dtype,
                eredu_checkpoint::schema::StoredDtypeConstraint::Exact(
                    eredu_checkpoint::StoredDtype::I32
                )
            ) {
                (
                    Dtype::I32,
                    (0..elements).flat_map(|_| 0_i32.to_le_bytes()).collect(),
                )
            } else if constraint.key.ends_with(".A_log") {
                (
                    Dtype::F32,
                    (-1.0_f32)
                        .to_le_bytes()
                        .into_iter()
                        .cycle()
                        .take(elements * 4)
                        .collect(),
                )
            } else if constraint.key.contains("norm") && constraint.key.ends_with(".weight") {
                (
                    Dtype::F32,
                    (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                )
            } else if constraint.role == eredu_checkpoint::schema::TensorRole::Companion {
                (
                    Dtype::F32,
                    (0..elements).flat_map(|_| 1.0_f32.to_le_bytes()).collect(),
                )
            } else {
                let seed = constraint.key.bytes().fold(1_u32, |value, byte| {
                    value.wrapping_mul(31) ^ u32::from(byte)
                });
                (
                    Dtype::F32,
                    (0..elements)
                        .flat_map(|index| {
                            let signed = i32::try_from((seed as usize + index) % 29).unwrap() - 14;
                            (signed as f32 * 0.001).to_le_bytes()
                        })
                        .collect(),
                )
            };
            (
                constraint.key.clone(),
                constraint.shape.clone(),
                dtype,
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, dtype, bytes)| {
            (
                name.as_str(),
                TensorView::new(*dtype, shape.clone(), bytes.as_slice()).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &root.path().join("model.safetensors")).unwrap();
    root
}

fn tiny_heterogeneous_gguf(
    family: &str,
    stream: &Stream,
) -> crate::tests::support::test_utils::SyntheticGguf {
    tiny_heterogeneous_gguf_with_packed_qwen_next(family, None, stream)
}

fn tiny_heterogeneous_gguf_with_packed_qwen_next(
    family: &str,
    packed_qkvz: Option<eredu_gguf::GgmlType>,
    stream: &Stream,
) -> crate::tests::support::test_utils::SyntheticGguf {
    use std::collections::HashMap;

    use eredu_gguf::{MetadataArray, MetadataValue};

    let (plan, metadata) = match family {
        "lfm2" => {
            let args =
                eredu_architectures::lfm2::model_args_from_config_value(&lfm2_config()).unwrap();
            let key = |suffix: &str| format!("lfm2.{suffix}");
            (
                eredu_architectures::lfm2::gguf_plan(&args).unwrap(),
                HashMap::from([
                    (
                        "general.architecture".into(),
                        MetadataValue::String("lfm2".into()),
                    ),
                    ("general.file_type".into(), MetadataValue::Uint32(0)),
                    (key("block_count"), MetadataValue::Uint32(2)),
                    (key("embedding_length"), MetadataValue::Uint32(16)),
                    (
                        key("feed_forward_length"),
                        MetadataValue::Uint32(args.dense_intermediate_size as u32),
                    ),
                    (key("attention.head_count"), MetadataValue::Uint32(4)),
                    (
                        key("attention.head_count_kv"),
                        MetadataValue::Array(MetadataArray::Uint32(vec![0, 2])),
                    ),
                    (
                        key("attention.layer_norm_rms_epsilon"),
                        MetadataValue::Float32(args.norm_eps),
                    ),
                    (key("context_length"), MetadataValue::Uint32(64)),
                    (key("shortconv.l_cache"), MetadataValue::Uint32(3)),
                    (
                        key("rope.freq_base"),
                        MetadataValue::Float32(args.rope.theta),
                    ),
                    (key("vocab_size"), MetadataValue::Uint32(64)),
                ]),
            )
        }
        "kimi_linear" => {
            let args = eredu_architectures::kimi_linear::model_args_from_config_value(
                &kimi_linear_config(),
            )
            .unwrap();
            let key = |suffix: &str| format!("kimi-linear.{suffix}");
            (
                eredu_architectures::kimi_linear::gguf_plan(&args).unwrap(),
                HashMap::from([
                    (
                        "general.architecture".into(),
                        MetadataValue::String("kimi-linear".into()),
                    ),
                    ("general.file_type".into(), MetadataValue::Uint32(0)),
                    (key("block_count"), MetadataValue::Uint32(2)),
                    (key("embedding_length"), MetadataValue::Uint32(12)),
                    (key("attention.head_count"), MetadataValue::Uint32(3)),
                    (
                        key("attention.head_count_kv"),
                        MetadataValue::Array(MetadataArray::Uint32(vec![0, 1])),
                    ),
                    (key("rope.dimension_count"), MetadataValue::Uint32(2)),
                    (key("attention.key_length_mla"), MetadataValue::Uint32(6)),
                    (key("vocab_size"), MetadataValue::Uint32(64)),
                    (key("feed_forward_length"), MetadataValue::Uint32(16)),
                    (key("context_length"), MetadataValue::Uint32(64)),
                    (
                        key("attention.layer_norm_rms_epsilon"),
                        MetadataValue::Float32(args.rms_norm_eps),
                    ),
                    (key("kda.head_dim"), MetadataValue::Uint32(4)),
                    (key("ssm.conv_kernel"), MetadataValue::Uint32(3)),
                    (key("expert_count"), MetadataValue::Uint32(2)),
                    (key("expert_feed_forward_length"), MetadataValue::Uint32(8)),
                    (key("attention.kv_lora_rank"), MetadataValue::Uint32(6)),
                    (key("attention.value_length_mla"), MetadataValue::Uint32(4)),
                    (key("leading_dense_block_count"), MetadataValue::Uint32(2)),
                    (key("expert_used_count"), MetadataValue::Uint32(1)),
                    (key("expert_shared_count"), MetadataValue::Uint32(1)),
                ]),
            )
        }
        "nemotron_h" => {
            let args =
                eredu_architectures::nemotron_h::model_args_from_config_value(&nemotron_h_config())
                    .unwrap();
            let key = |suffix: &str| format!("nemotron_h.{suffix}");
            (
                eredu_architectures::nemotron_h::gguf_plan(&args).unwrap(),
                HashMap::from([
                    (
                        "general.architecture".into(),
                        MetadataValue::String("nemotron_h".into()),
                    ),
                    ("general.file_type".into(), MetadataValue::Uint32(0)),
                    (key("block_count"), MetadataValue::Uint32(4)),
                    (key("embedding_length"), MetadataValue::Uint32(16)),
                    (
                        key("feed_forward_length"),
                        MetadataValue::Array(MetadataArray::Uint32(vec![0, 0, 24, 0])),
                    ),
                    (
                        key("attention.head_count_kv"),
                        MetadataValue::Array(MetadataArray::Uint32(vec![0, 2, 0, 0])),
                    ),
                    (key("attention.head_count"), MetadataValue::Uint32(4)),
                    (key("attention.key_length"), MetadataValue::Uint32(4)),
                    (
                        key("attention.layer_norm_rms_epsilon"),
                        MetadataValue::Float32(args.norm_eps),
                    ),
                    (key("context_length"), MetadataValue::Uint32(64)),
                    (key("ssm.inner_size"), MetadataValue::Uint32(16)),
                    (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
                    (key("ssm.state_size"), MetadataValue::Uint32(3)),
                    (key("ssm.group_count"), MetadataValue::Uint32(2)),
                    (key("ssm.conv_kernel"), MetadataValue::Uint32(3)),
                    (key("vocab_size"), MetadataValue::Uint32(64)),
                ]),
            )
        }
        "qwen35" | "qwen3next" => {
            let (config, architecture) = if family == "qwen35" {
                (qwen_hybrid_config(), "qwen35")
            } else {
                (qwen_next_config(), "qwen3next")
            };
            let args = eredu_architectures::qwen::hybrid::model_args_from_config_value(&config)
                .unwrap()
                .text;
            let key = |suffix: &str| format!("{architecture}.{suffix}");
            let plan = eredu_architectures::qwen::hybrid::gguf_plan(&args).unwrap();
            let metadata = HashMap::from([
                (
                    "general.architecture".into(),
                    MetadataValue::String(architecture.into()),
                ),
                ("general.file_type".into(), MetadataValue::Uint32(0)),
                (key("block_count"), MetadataValue::Uint32(2)),
                (key("embedding_length"), MetadataValue::Uint32(32)),
                (key("attention.head_count"), MetadataValue::Uint32(4)),
                (key("attention.head_count_kv"), MetadataValue::Uint32(2)),
                (key("attention.key_length"), MetadataValue::Uint32(8)),
                (key("rope.dimension_count"), MetadataValue::Uint32(2)),
                (key("full_attention_interval"), MetadataValue::Uint32(2)),
                (key("vocab_size"), MetadataValue::Uint32(64)),
                (key("context_length"), MetadataValue::Uint32(128)),
                (
                    key("attention.layer_norm_rms_epsilon"),
                    MetadataValue::Float32(args.rms_norm_eps),
                ),
                (key("feed_forward_length"), MetadataValue::Uint32(48)),
                (key("ssm.conv_kernel"), MetadataValue::Uint32(4)),
                (key("ssm.state_size"), MetadataValue::Uint32(8)),
                (key("ssm.group_count"), MetadataValue::Uint32(2)),
                (key("ssm.time_step_rank"), MetadataValue::Uint32(4)),
            ]);
            (plan, metadata)
        }
        _ => unreachable!("heterogeneous GGUF fixture family"),
    };
    let mut constraints = plan.common_tensors.iter().collect::<Vec<_>>();
    constraints.extend(
        plan.layout_groups
            .iter()
            .filter(|group| group.required)
            .filter_map(|group| {
                if family == "qwen3next"
                    && group.variants.iter().any(|variant| variant.id == "fused")
                {
                    group.variants.iter().find(|variant| variant.id == "fused")
                } else {
                    group.variants.first()
                }
            })
            .flat_map(|variant| variant.tensors.iter()),
    );
    let arrays = constraints
        .into_iter()
        .filter(|constraint| {
            constraint.requirement == eredu_checkpoint::schema::TensorRequirement::Required
        })
        .map(|constraint| {
            let shape = constraint
                .shape
                .iter()
                .map(|dimension| i32::try_from(*dimension).unwrap())
                .collect::<Vec<_>>();
            let array = if constraint.key.ends_with("ssm_a") {
                Array::full::<f32>(&shape, Array::from_f32(-1.0), stream).unwrap()
            } else {
                let seed = constraint
                    .key
                    .bytes()
                    .fold(0_usize, |sum, byte| sum.wrapping_add(usize::from(byte)));
                let values = (0..shape.iter().map(|dimension| *dimension as usize).product())
                    .map(|index| ((index + seed) % 29 + 1) as f32 / 100.0)
                    .collect::<Vec<_>>();
                Array::from_slice(&values, &shape)
            };
            (constraint.key.clone(), array)
        })
        .collect::<HashMap<_, _>>();
    crate::tests::support::test_utils::SyntheticGguf::with_packed_tensors(
        &arrays,
        &metadata,
        |name, _| packed_qkvz.filter(|_| name.contains("attn_qkvz.weight")),
    )
}

fn lfm2_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "lfm2", "vocab_size": 64, "hidden_size": 16,
        "intermediate_size": 32, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": 64,
        "layer_types": ["conv", "full_attention"], "conv_L_cache": 3,
        "block_multiple_of": 8, "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": true, "tie_word_embeddings": false
    })
}

fn routed_lfm2_config() -> serde_json::Value {
    let mut config = lfm2_config();
    config["model_type"] = "lfm2_moe".into();
    config["num_dense_layers"] = 1.into();
    config["moe_intermediate_size"] = 8.into();
    config["num_experts"] = 2.into();
    config["num_experts_per_tok"] = 1.into();
    config
}

fn kimi_linear_config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"kimi_linear","vocab_size":64,"hidden_size":12,"num_hidden_layers":2,
        "num_attention_heads":3,"num_key_value_heads":3,"intermediate_size":16,"head_dim":4,
        "model_max_length":64,"linear_attn_config":{"kda_layers":[1],"full_attn_layers":[2],"num_heads":3,"head_dim":4,"short_conv_kernel_size":3},
        "num_experts":2,"moe_intermediate_size":8,"kv_lora_rank":6,"qk_nope_head_dim":4,"qk_rope_head_dim":2,"v_head_dim":4,
        "mla_use_nope":true,"num_experts_per_token":1,"num_shared_experts":1,"routed_scaling_factor":1.0,
        "first_k_dense_replace":2,"num_expert_group":1,"topk_group":1
    })
}

fn routed_kimi_linear_config() -> serde_json::Value {
    let mut config = kimi_linear_config();
    config["first_k_dense_replace"] = 1.into();
    config
}

fn nemotron_h_config() -> serde_json::Value {
    serde_json::json!({
        "model_type":"nemotron_h", "vocab_size":64, "hidden_size":16,
        "intermediate_size":24, "num_hidden_layers":4,
        "hybrid_override_pattern":"M*-M", "num_attention_heads":4,
        "num_key_value_heads":2, "head_dim":4, "mamba_num_heads":4,
        "n_groups":2, "mamba_head_dim":4, "ssm_state_size":3,
        "conv_kernel":3, "n_routed_experts":4, "n_shared_experts":1,
        "moe_intermediate_size":8, "moe_shared_expert_intermediate_size":8,
        "num_experts_per_tok":2, "n_group":2, "topk_group":1,
        "num_nextn_predict_layers":0
    })
}

fn packed_alias_nemotron_h_config() -> serde_json::Value {
    let mut config = nemotron_h_config();
    config["hidden_size"] = 32.into();
    config["intermediate_size"] = 32.into();
    config["head_dim"] = 8.into();
    config["mamba_head_dim"] = 8.into();
    config["moe_intermediate_size"] = 32.into();
    config["moe_shared_expert_intermediate_size"] = 32.into();
    config["quantization"] = serde_json::json!({ "group_size": 32, "bits": 4 });
    config
}

fn qwen_hybrid_config() -> serde_json::Value {
    serde_json::json!({
        "model_type": "qwen3_5_text", "vocab_size": 64, "hidden_size": 32,
        "num_hidden_layers": 2, "mtp_num_hidden_layers": 0,
        "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 8,
        "max_position_embeddings": 128, "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 8, "linear_value_head_dim": 8,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4,
        "intermediate_size": 48, "moe_intermediate_size": 16,
        "shared_expert_intermediate_size": 24, "num_experts_per_tok": 0,
        "num_experts": 0, "layer_types": ["linear_attention", "full_attention"]
    })
}

fn routed_qwen_hybrid_config() -> serde_json::Value {
    let mut config = qwen_hybrid_config();
    config["model_type"] = "qwen3_5_moe_text".into();
    config["num_experts"] = 2.into();
    config["num_experts_per_tok"] = 1.into();
    config
}

fn routed_qwen_next_config() -> serde_json::Value {
    let mut config = routed_qwen_hybrid_config();
    config["model_type"] = "qwen3_next".into();
    config
}

fn routed_deepseek_v3_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3", "hidden_size": 16,
        "intermediate_size": 24, "moe_intermediate_size": 8,
        "num_hidden_layers": 2, "num_nextn_predict_layers": 0,
        "num_attention_heads": 2, "vocab_size": 64,
        "max_position_embeddings": 64, "q_lora_rank": 4,
        "kv_lora_rank": 4, "qk_nope_head_dim": 6,
        "qk_rope_head_dim": 2, "v_head_dim": 8,
        "first_k_dense_replace": 1, "moe_layer_freq": 1,
        "n_routed_experts": 2, "n_shared_experts": 1,
        "num_experts_per_tok": 1, "n_group": 1, "topk_group": 1,
        "topk_method": "noaux_tc", "scoring_func": "sigmoid",
        "norm_topk_prob": true, "routed_scaling_factor": 1.0,
        "tie_word_embeddings": false
    })
}

fn routed_deepseek_v4_config() -> serde_json::Value {
    serde_json::json!({
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4", "hidden_size": 8,
        "moe_intermediate_size": 4, "num_hidden_layers": 3,
        "num_nextn_predict_layers": 0, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 4, "qk_rope_head_dim": 2,
        "q_lora_rank": 2, "o_lora_rank": 2, "o_groups": 2,
        "vocab_size": 64, "max_position_embeddings": 128,
        "sliding_window": 4, "compress_ratios": [0, 4, 0],
        "index_n_heads": 2, "index_head_dim": 4, "index_topk": 1,
        "hc_mult": 2, "hc_sinkhorn_iters": 2,
        "n_routed_experts": 2, "n_shared_experts": 1,
        "num_experts_per_tok": 1, "num_hash_layers": 1,
        "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "norm_topk_prob": true, "routed_scaling_factor": 1.0,
        "swiglu_limit": 4.0, "tie_word_embeddings": false
    })
}

fn qwen_next_config() -> serde_json::Value {
    let mut config = qwen_hybrid_config();
    config["model_type"] = "qwen3_next".into();
    config
}

fn execution_streams() -> (Stream, Stream) {
    let execution_device = if cfg!(feature = "metal") {
        safemlx::Device::new(safemlx::DeviceType::Gpu, 0)
    } else {
        safemlx::Device::new(safemlx::DeviceType::Cpu, 0)
    };
    let weights_device = safemlx::Device::new(safemlx::DeviceType::Cpu, 0);
    (
        Stream::new_with_device(&execution_device),
        Stream::new_with_device(&weights_device),
    )
}

fn complete_state_capabilities(
    components: impl IntoIterator<Item = StateComponentMechanism>,
) -> StateMechanismCapabilities {
    StateMechanismCapabilities::new(components)
        .with_transactions(true, true)
        .with_reset(true)
        .with_prompt_cache(true)
        .with_observation_retention(true)
}

fn capabilities_with(
    full: &BackendMechanismCapabilities,
    operators: eredu_nn::NeuralOperatorCapabilities,
    state: StateMechanismCapabilities,
) -> BackendMechanismCapabilities {
    let state = match full.state().floating_state_dtype() {
        Some((source, dtype)) => state.with_floating_state_dtype(source.clone(), dtype),
        None => state,
    };
    BackendMechanismCapabilities::new(
        operators,
        full.weight_lowerings().to_vec(),
        full.weight_residencies().to_vec(),
        state,
    )
    .with_session(full.session())
    .with_prompt_cache(full.prompt_cache())
    .with_exact_completion(full.exact_completion())
    .with_grouped_operations(full.grouped_operations().iter().copied())
}

fn tiny_llama_gguf(
    architecture: &str,
    packed: Option<eredu_gguf::GgmlType>,
    stream: &Stream,
) -> crate::tests::support::test_utils::SyntheticGguf {
    use std::collections::HashMap;

    use eredu_gguf::MetadataValue;

    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), MetadataValue::Uint32(0)),
        (key("block_count"), MetadataValue::Uint32(1)),
        (key("embedding_length"), MetadataValue::Uint32(32)),
        (key("attention.head_count"), MetadataValue::Uint32(4)),
        (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
        (key("feed_forward_length"), MetadataValue::Uint32(64)),
        (
            key("attention.layer_norm_rms_epsilon"),
            MetadataValue::Float32(1e-5),
        ),
        (key("vocab_size"), MetadataValue::Uint32(64)),
        (key("context_length"), MetadataValue::Uint32(32)),
        (key("rope.freq_base"), MetadataValue::Float32(10_000.0)),
    ]);
    let tensors = [
        ("token_embd.weight", vec![64, 32]),
        ("output_norm.weight", vec![32]),
        ("blk.0.attn_norm.weight", vec![32]),
        ("blk.0.ffn_norm.weight", vec![32]),
        ("blk.0.attn_q.weight", vec![32, 32]),
        ("blk.0.attn_k.weight", vec![8, 32]),
        ("blk.0.attn_v.weight", vec![8, 32]),
        ("blk.0.attn_output.weight", vec![32, 32]),
        ("blk.0.ffn_gate.weight", vec![64, 32]),
        ("blk.0.ffn_up.weight", vec![64, 32]),
        ("blk.0.ffn_down.weight", vec![32, 64]),
    ]
    .into_iter()
    .map(|(name, shape)| {
        (
            name.to_string(),
            Array::zeros::<f32>(&shape, stream).unwrap(),
        )
    })
    .collect::<HashMap<_, _>>();
    crate::tests::support::test_utils::SyntheticGguf::with_packed_tensors(
        &tensors,
        &metadata,
        |name, array| packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2),
    )
}

fn tiny_qwen_gguf(
    architecture: &str,
    packed: Option<eredu_gguf::GgmlType>,
    stream: &Stream,
) -> crate::tests::support::test_utils::SyntheticGguf {
    use std::collections::HashMap;

    use eredu_gguf::MetadataValue;

    let key = |suffix: &str| format!("{architecture}.{suffix}");
    let metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetadataValue::String(architecture.into()),
        ),
        ("general.file_type".into(), MetadataValue::Uint32(0)),
        (key("block_count"), MetadataValue::Uint32(1)),
        (key("embedding_length"), MetadataValue::Uint32(32)),
        (key("attention.head_count"), MetadataValue::Uint32(4)),
        (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
        (key("feed_forward_length"), MetadataValue::Uint32(64)),
        (
            key("attention.layer_norm_rms_epsilon"),
            MetadataValue::Float32(1e-5),
        ),
        (key("vocab_size"), MetadataValue::Uint32(64)),
        (key("context_length"), MetadataValue::Uint32(32)),
        (key("rope.freq_base"), MetadataValue::Float32(1_000_000.0)),
    ]);
    let mut tensors = vec![
        ("token_embd.weight", vec![64, 32]),
        ("output_norm.weight", vec![32]),
        ("blk.0.attn_norm.weight", vec![32]),
        ("blk.0.ffn_norm.weight", vec![32]),
        ("blk.0.attn_q.weight", vec![32, 32]),
        ("blk.0.attn_k.weight", vec![8, 32]),
        ("blk.0.attn_v.weight", vec![8, 32]),
        ("blk.0.attn_output.weight", vec![32, 32]),
        ("blk.0.ffn_gate.weight", vec![64, 32]),
        ("blk.0.ffn_up.weight", vec![64, 32]),
        ("blk.0.ffn_down.weight", vec![32, 64]),
    ];
    if architecture == "qwen2" {
        tensors.extend([
            ("blk.0.attn_q.bias", vec![32]),
            ("blk.0.attn_k.bias", vec![8]),
            ("blk.0.attn_v.bias", vec![8]),
        ]);
    } else {
        tensors.extend([
            ("blk.0.attn_q_norm.weight", vec![8]),
            ("blk.0.attn_k_norm.weight", vec![8]),
        ]);
    }
    let tensors = tensors
        .into_iter()
        .map(|(name, shape)| {
            (
                name.to_string(),
                Array::zeros::<f32>(&shape, stream).unwrap(),
            )
        })
        .collect::<HashMap<_, _>>();
    crate::tests::support::test_utils::SyntheticGguf::with_packed_tensors(
        &tensors,
        &metadata,
        |name, array| packed.filter(|_| name.ends_with(".weight") && array.ndim() == 2),
    )
}

fn tiny_qwen_moe_gguf(stream: &Stream) -> crate::tests::support::test_utils::SyntheticGguf {
    use std::collections::HashMap;

    use eredu_gguf::MetadataValue;

    let key = |suffix: &str| format!("qwen3moe.{suffix}");
    let metadata = HashMap::from([
        (
            "general.architecture".into(),
            MetadataValue::String("qwen3moe".into()),
        ),
        ("general.file_type".into(), MetadataValue::Uint32(0)),
        (key("block_count"), MetadataValue::Uint32(1)),
        (key("embedding_length"), MetadataValue::Uint32(32)),
        (key("attention.head_count"), MetadataValue::Uint32(4)),
        (key("attention.head_count_kv"), MetadataValue::Uint32(1)),
        (key("attention.key_length"), MetadataValue::Uint32(8)),
        (
            key("attention.layer_norm_rms_epsilon"),
            MetadataValue::Float32(1e-5),
        ),
        (key("feed_forward_length"), MetadataValue::Uint32(64)),
        (key("expert_feed_forward_length"), MetadataValue::Uint32(16)),
        (key("expert_count"), MetadataValue::Uint32(2)),
        (key("expert_used_count"), MetadataValue::Uint32(1)),
        (key("vocab_size"), MetadataValue::Uint32(64)),
        (key("context_length"), MetadataValue::Uint32(32)),
        (key("rope.freq_base"), MetadataValue::Float32(10_000.0)),
    ]);
    let tensors = [
        ("token_embd.weight", vec![64, 32]),
        ("output_norm.weight", vec![32]),
        ("blk.0.attn_norm.weight", vec![32]),
        ("blk.0.ffn_norm.weight", vec![32]),
        ("blk.0.attn_q.weight", vec![32, 32]),
        ("blk.0.attn_k.weight", vec![8, 32]),
        ("blk.0.attn_v.weight", vec![8, 32]),
        ("blk.0.attn_output.weight", vec![32, 32]),
        ("blk.0.attn_q_norm.weight", vec![8]),
        ("blk.0.attn_k_norm.weight", vec![8]),
        ("blk.0.ffn_gate_inp.weight", vec![2, 32]),
        ("blk.0.ffn_gate_exps.weight", vec![2, 16, 32]),
        ("blk.0.ffn_up_exps.weight", vec![2, 16, 32]),
        ("blk.0.ffn_down_exps.weight", vec![2, 32, 16]),
    ]
    .into_iter()
    .map(|(name, shape)| {
        (
            name.to_owned(),
            Array::zeros::<f32>(&shape, stream).unwrap(),
        )
    })
    .collect::<HashMap<_, _>>();
    crate::tests::support::test_utils::SyntheticGguf::with_packed_tensors(
        &tensors,
        &metadata,
        |_, _| None,
    )
}
