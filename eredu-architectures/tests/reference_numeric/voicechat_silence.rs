use super::*;
use eredu_architectures::voicechat::{
    config::{PreprocessingConfig, VoiceChatConfig},
    conformer::ConformerDimensions,
    perception::PerceptionDimensions,
    perception_rnnt::PerceptionRnnt,
    rnnt::RnntDimensions,
    silence::{PreparedSilenceConditioning, SilenceConditioningError},
    subsampling::SubsamplingDimensions,
};
use eredu_nn::TensorElementType;
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(eredu_evaluation::voicechat::TOOL_SILENCE_REFERENCE_JSON).unwrap()
}

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect(),
    )
}

fn config() -> PreprocessingConfig {
    VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap()
    .perception
    .preprocessor
}

fn prepared(fixture: &Value) -> PreparedSilenceConditioning {
    PreparedSilenceConditioning::with_coefficients(
        &config(),
        tensor(&fixture["preprocessing"]["window"]).data,
        tensor(&fixture["preprocessing"]["filters"]).data,
    )
    .unwrap()
}

fn dimensions(fixture: &Value) -> (PerceptionDimensions, RnntDimensions) {
    let integer = |component: &str, field: &str| {
        fixture[component]["geometry"][field].as_i64().unwrap() as i32
    };
    let width = integer("conformer", "width");
    (
        PerceptionDimensions {
            subsampling: SubsamplingDimensions {
                features: integer("subsampling", "features"),
                channels: integer("subsampling", "channels"),
                width: integer("subsampling", "width"),
            },
            block: ConformerDimensions {
                width,
                heads: integer("conformer", "heads"),
                ff_hidden: integer("conformer", "feed_forward"),
                conv_kernel: integer("conformer", "kernel"),
                left_context: integer("conformer", "left_context"),
            },
            layers: 1,
            output_width: fixture["perception"]["output_width"].as_i64().unwrap() as i32,
        },
        RnntDimensions {
            vocab_size: 1,
            predictor_width: 1,
            predictor_layers: 1,
            encoder_width: width,
            joint_width: 1,
        },
    )
}

fn model(fixture: &Value, context: &NumericContext) -> PerceptionRnnt<NumericBackend> {
    struct Bind(BTreeMap<String, Value>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut NumericTensor) {
            let name = metadata.id.as_str();
            if name.starts_with("stt_model.rnnt_") {
                // Startup conditioning must not execute transcription. Its RNNT
                // slots deliberately cannot satisfy even an embedding lookup.
                *slot = NumericTensor::zeros(vec![0]);
                return;
            }
            let value = tensor(self.0.get(name).unwrap_or_else(|| panic!("missing {name}")));
            assert_eq!(slot.shape(), value.shape(), "{name}");
            *slot = value;
        }
    }
    let parameters = ["subsampling", "conformer", "perception"]
        .into_iter()
        .flat_map(|part| {
            fixture[part]["parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| (name.clone(), value.clone()))
        })
        .collect();
    let (perception, rnnt) = dimensions(fixture);
    let mut model = PerceptionRnnt::new(perception, rnnt, context).unwrap();
    model.visit_parameters_mut(&mut Bind(parameters));
    model
}

fn assert_tensor(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (index, (actual, expected)) in actual.data.iter().zip(expected.data).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "silence element {index}: {actual} != NVIDIA {expected}"
        );
    }
}

#[test]
fn voicechat_startup_silence_full_perception_and_bf16_match_pinned_nvidia() {
    let fixture = fixture();
    let context = NumericContext::default();
    let prepared = prepared(&fixture);
    let mel = prepared.mel_features().unwrap();
    assert_eq!((mel.frames, mel.mel_bins), (101, 128));
    let mel = NumericTensor::new(vec![1, 101, 128], mel.values);
    assert_tensor(&mel, &fixture["preprocessing"]["mel"], &fixture);
    let mut model = model(&fixture, &context);
    let first = model
        .prepare_silence(&prepared, TensorElementType::F32, &context)
        .unwrap();
    assert_eq!(first.element_type(), Some(TensorElementType::F32));
    assert!(first.data.iter().any(|value| value.abs() > 0.001));
    assert_tensor(&first, &fixture["perception"]["embedding"], &fixture);
    let reduced = model
        .prepare_silence(&prepared, TensorElementType::Bf16, &context)
        .unwrap();
    assert_eq!(reduced.element_type(), Some(TensorElementType::Bf16));
    assert_eq!(reduced.shape(), &[1, 1, 6]);
    assert_eq!(
        reduced.data,
        tensor(&fixture["perception"]["bf16_embedding"]).data
    );
    assert_ne!(
        first.data, reduced.data,
        "final conditioning uses the selected backbone precision"
    );
}

#[test]
fn voicechat_startup_silence_rejects_invalid_coefficients_precision_and_perception_geometry() {
    let fixture = fixture();
    let context = NumericContext::default();
    let filters = tensor(&fixture["preprocessing"]["filters"]).data;
    assert!(matches!(
        PreparedSilenceConditioning::with_coefficients(&config(), vec![1.; 399], filters),
        Err(SilenceConditioningError::Geometry(_))
    ));
    let prepared = prepared(&fixture);
    let mut model = model(&fixture, &context);
    assert!(matches!(
        model.prepare_silence(&prepared, TensorElementType::U32, &context),
        Err(SilenceConditioningError::Geometry(_))
    ));
    let (mut perception, rnnt) = dimensions(&fixture);
    perception.subsampling.features = 8;
    let mut wrong_width =
        PerceptionRnnt::<NumericBackend>::new(perception, rnnt, &context).unwrap();
    assert!(matches!(
        wrong_width.prepare_silence(&prepared, TensorElementType::F32, &context),
        Err(SilenceConditioningError::Geometry(_))
    ));
}
