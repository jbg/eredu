//! Tool silence uses the released full-perception startup path on native CPU.
#![cfg(feature = "audio")]
use eredu_architectures::voicechat::{
    config::VoiceChatConfig,
    conformer::ConformerDimensions,
    perception::{Perception, PerceptionDimensions},
    silence::PreparedSilenceConditioning,
    subsampling::SubsamplingDimensions,
};
use eredu_backend_mlx::{backend::nn::shared::MlxNeuralBackend as Backend, MlxTensor};
use eredu_nn::{ParameterMetadata, ParameterVisitorMut, Parameterized, Tensor, TensorElementType};
use safemlx::{Array, Device, DeviceType, Stream};
use serde_json::Value;
use std::collections::BTreeMap;

fn values(value: &Value) -> Vec<f32> {
    value["values"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}

#[test]
fn cpu_startup_silence_matches_nvidia_full_perception_and_backbone_precision() {
    let fixture: Value =
        serde_json::from_str(eredu_evaluation::voicechat::TOOL_SILENCE_REFERENCE_JSON).unwrap();
    let stream = Stream::new_with_device(&Device::new(DeviceType::Cpu, 0));
    let config = VoiceChatConfig::from_json(include_str!(
        "../../eredu-architectures/src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let prepared = PreparedSilenceConditioning::with_coefficients(
        &config.perception.preprocessor,
        values(&fixture["preprocessing"]["window"]),
        values(&fixture["preprocessing"]["filters"]),
    )
    .unwrap();
    let geometry =
        |part: &str, name: &str| fixture[part]["geometry"][name].as_i64().unwrap() as i32;
    let mut model = Perception::<Backend>::new(
        PerceptionDimensions {
            subsampling: SubsamplingDimensions {
                features: geometry("subsampling", "features"),
                channels: geometry("subsampling", "channels"),
                width: geometry("subsampling", "width"),
            },
            block: ConformerDimensions {
                width: geometry("conformer", "width"),
                heads: geometry("conformer", "heads"),
                ff_hidden: geometry("conformer", "feed_forward"),
                conv_kernel: geometry("conformer", "kernel"),
                left_context: geometry("conformer", "left_context"),
            },
            layers: 1,
            output_width: fixture["perception"]["output_width"].as_i64().unwrap() as i32,
        },
        &stream,
    )
    .unwrap();
    struct Bind(BTreeMap<String, Value>);
    impl<'a> ParameterVisitorMut<'a, MlxTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut MlxTensor) {
            let value = &self.0[metadata.id.as_str()];
            let shape: Vec<i32> = value["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            assert_eq!(slot.shape(), shape);
            *slot = MlxTensor::from_array(Array::from_slice(&values(value), &shape));
        }
    }
    model.visit_parameters_mut(&mut Bind(
        ["subsampling", "conformer", "perception"]
            .into_iter()
            .flat_map(|part| {
                fixture[part]["parameters"]
                    .as_object()
                    .unwrap()
                    .iter()
                    .map(|(name, value)| (name.clone(), value.clone()))
            })
            .collect(),
    ));
    let full = prepared
        .execute(&mut model, TensorElementType::F32, &stream)
        .unwrap();
    assert_eq!(full.shape(), &[1, 1, 6]);
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (actual, expected) in full
        .to_f32_vec(&stream)
        .unwrap()
        .into_iter()
        .zip(values(&fixture["perception"]["embedding"]))
    {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "{actual} versus NVIDIA {expected}"
        );
    }
    let reduced = prepared
        .execute(&mut model, TensorElementType::Bf16, &stream)
        .unwrap();
    assert_eq!(reduced.element_type(), Some(TensorElementType::Bf16));
    assert_eq!(
        reduced.to_f32_vec(&stream).unwrap(),
        values(&fixture["perception"]["bf16_embedding"])
    );
}
