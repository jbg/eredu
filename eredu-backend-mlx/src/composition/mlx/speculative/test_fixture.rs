//! Compact assistant input for shared native submission-recovery conformance.
use eredu_architectures::gemma4;
use eredu_checkpoint::schema::StoredDtypeConstraint;
use safetensors::tensor::{serialize_to_file, Dtype, TensorView};

pub(super) const ASSISTANT_CONFIG: &str = r#"{
  "model_type":"gemma4_assistant","backbone_hidden_size":32,
  "use_ordered_embeddings":false,"tie_word_embeddings":false,"block_size":4,
  "text_config":{"model_type":"gemma4_text","hidden_size":32,
    "num_hidden_layers":1,"intermediate_size":64,"num_attention_heads":4,
    "num_key_value_heads":2,"head_dim":8,"rms_norm_eps":0.00001,
    "vocab_size":32,"max_position_embeddings":128,"tie_word_embeddings":false,
    "attention_k_eq_v":false,"layer_types":["full_attention"]}
}"#;

pub(super) fn assistant_artifact() -> tempfile::TempDir {
    assistant_artifact_values(false)
}

fn assistant_artifact_values(nonzero: bool) -> tempfile::TempDir {
    let directory = tempfile::tempdir().unwrap();
    std::fs::write(directory.path().join("config.json"), ASSISTANT_CONFIG).unwrap();
    let config = gemma4::AssistantConfig::from_json(ASSISTANT_CONFIG.as_bytes()).unwrap();
    let plan = gemma4::assistant_safetensors_plan(&config).unwrap();
    assert!(plan.layout_groups.is_empty());
    let tensors = plan
        .common_tensors
        .iter()
        .enumerate()
        .map(|(index, tensor)| {
            assert_eq!(tensor.dtype, StoredDtypeConstraint::Floating);
            let bytes = (0..tensor.shape.iter().product::<usize>())
                .flat_map(|element| {
                    let value = if nonzero {
                        (index * 10000 + element + 1) as f32 / 1024.0
                    } else {
                        0.0
                    };
                    value.to_le_bytes()
                })
                .collect::<Vec<_>>();
            (tensor.key.clone(), tensor.shape.clone(), bytes)
        })
        .collect::<Vec<_>>();
    let views = tensors
        .iter()
        .map(|(name, shape, bytes)| {
            (
                name.as_str(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            )
        })
        .collect::<Vec<_>>();
    serialize_to_file(views, None, &directory.path().join("model.safetensors")).unwrap();
    directory
}
