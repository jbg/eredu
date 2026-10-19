//! Compact artifact fixture for shared native session conformance.
use super::*;
use eredu_core::ModelConfigurationResolver;

pub(crate) fn tiny_artifact(model_type: &str, tied: bool) -> tempfile::TempDir {
    tiny_safetensors_artifact(model_type, tied, false, false)
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
