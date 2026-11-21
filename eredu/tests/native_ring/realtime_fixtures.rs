use eredu_checkpoint::{schema::StoredDtypeConstraint, StoredDtype};
use safetensors::tensor::{serialize_to_file, Dtype as SafeDtype, TensorView};
use std::path::Path;

const TINY_NATIVE_CONFIG: &str = r#"{
    "model_type": "moshi",
    "dim": 32,
    "text_card": 32,
    "n_q": 2,
    "dep_q": 1,
    "generated_audio_codebooks": 1,
    "card": 32,
    "num_heads": 4,
    "num_layers": 1,
    "dim_feedforward": 48,
    "causal": true,
    "context": 7,
    "max_period": 10000.0,
    "positional_embedding": "rope",
    "depformer_dim": 32,
    "depformer_dim_feedforward": 48,
    "depformer_num_heads": 4,
    "depformer_num_layers": 1,
    "depformer_context": 3,
    "depformer_max_period": 10000.0,
    "depformer_pos_emb": "none",
    "delays": [0, 0, 1]
}"#;

pub(super) fn write_tiny_native_artifact(directory: &Path) {
    let config_json = TINY_NATIVE_CONFIG;
    let config = eredu_architectures::moshi::MoshiConfig::from_json(config_json)
        .expect("tiny native Moshi config");
    let contract = eredu_architectures::moshi::parameter_contract(&config)
        .expect("tiny native parameter declaration");
    let plan = eredu_architectures::moshi::safetensors_plan(&contract)
        .expect("tiny native SafeTensors plan");
    assert!(plan.layout_groups.is_empty());

    // Derive every physical name and shape from the strict architecture
    // catalog. Zero matrices make greedy decisions exact, and unit
    // normalization scales remain valid.
    let tensors = plan
        .common_tensors
        .iter()
        .map(|constraint| {
            let dtype = match &constraint.dtype {
                StoredDtypeConstraint::Exact(dtype) => dtype.clone(),
                StoredDtypeConstraint::Floating => StoredDtype::F32,
                StoredDtypeConstraint::OneOf(dtypes) => dtypes
                    .iter()
                    .find(|dtype| **dtype == StoredDtype::F32)
                    .or_else(|| dtypes.first())
                    .cloned()
                    .expect("validated catalog dtype set"),
            };
            let elements = constraint.shape.iter().product::<usize>();
            let (dtype, bytes) = match dtype {
                StoredDtype::F32 => {
                    let value =
                        if constraint.key.contains("norm") || constraint.key.ends_with(".scales") {
                            1.0f32
                        } else {
                            0.0f32
                        };
                    (
                        SafeDtype::F32,
                        std::iter::repeat_n(value, elements)
                            .flat_map(f32::to_le_bytes)
                            .collect::<Vec<_>>(),
                    )
                }
                StoredDtype::U32 => (
                    SafeDtype::U32,
                    std::iter::repeat_n(0u32, elements)
                        .flat_map(u32::to_le_bytes)
                        .collect::<Vec<_>>(),
                ),
                StoredDtype::U8 => (
                    SafeDtype::U8,
                    vec![
                        if constraint.key.ends_with(".scales") {
                            127
                        } else {
                            0
                        };
                        elements
                    ],
                ),
                dtype => panic!("tiny native writer does not support {dtype:?}"),
            };
            (
                constraint.key.clone(),
                constraint.shape.clone(),
                dtype,
                bytes,
            )
        })
        .collect::<Vec<_>>();
    let views = tensors.iter().map(|(name, shape, dtype, bytes)| {
        (
            name.as_str(),
            TensorView::new(*dtype, shape.clone(), bytes).expect("catalog-derived tensor view"),
        )
    });
    std::fs::write(directory.join("config.json"), config_json).expect("write tiny native config");
    serialize_to_file(views, None, &directory.join("model.safetensors"))
        .expect("write tiny native SafeTensors artifact");
}
