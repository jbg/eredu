use super::*;
use eredu_architectures::voicechat::{
    conformer::ConformerDimensions,
    perception::PerceptionDimensions,
    perception_rnnt::{
        PerceptionRnnt, PerceptionRnntError, PerceptionRnntLimits, PerceptionRnntState,
    },
    rnnt::{RnntDimensions, RnntError, RnntLimits},
    subsampling::SubsamplingDimensions,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(eredu_evaluation::voicechat::PERCEPTION_REFERENCE_JSON).unwrap()
}

fn tensor(value: &Value) -> NumericTensor {
    NumericTensor::new(
        value["shape"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect::<Vec<_>>(),
        value["values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap() as f32)
            .collect(),
    )
}

fn dimensions() -> (PerceptionDimensions, RnntDimensions) {
    (
        PerceptionDimensions {
            subsampling: SubsamplingDimensions {
                features: 8,
                channels: 2,
                width: 4,
            },
            block: ConformerDimensions {
                width: 4,
                heads: 2,
                ff_hidden: 8,
                conv_kernel: 3,
                left_context: 3,
            },
            layers: 1,
            output_width: 6,
        },
        RnntDimensions {
            vocab_size: 1,
            predictor_width: 1,
            predictor_layers: 1,
            encoder_width: 4,
            joint_width: 1,
        },
    )
}

fn model(fixture: &Value, context: &NumericContext) -> PerceptionRnnt<NumericBackend> {
    struct Bind(BTreeMap<String, Value>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut NumericTensor) {
            let id = metadata.id.as_str();
            if let Some(value) = self.0.get(id) {
                let value = tensor(value);
                assert_eq!(slot.shape(), value.shape(), "{id}");
                *slot = value;
                return;
            }
            // An analytic RNNT decision boundary at encoder coordinate 1 = 1.12.
            // The independent NVIDIA perception fixture crosses this boundary
            // after its first frame; projection has a different width and values.
            let values = match id {
                "stt_model.rnnt_joint.enc.weight" => vec![0.0, 1.0, 0.0, 0.0],
                "stt_model.rnnt_joint.enc.bias" => vec![0.0],
                "stt_model.rnnt_joint.pred.weight" | "stt_model.rnnt_joint.pred.bias" => vec![0.0],
                "stt_model.rnnt_joint.joint_net.2.weight" => vec![1.0, 0.0],
                "stt_model.rnnt_joint.joint_net.2.bias" => vec![-1.12, 0.0],
                _ if id.starts_with("stt_model.rnnt_decoder.") => vec![0.1; slot.data.len()],
                _ => panic!("unexpected parameter {id}"),
            };
            *slot = NumericTensor::new(slot.shape(), values);
        }
    }
    let parameters = ["subsampling", "conformer", "perception"]
        .into_iter()
        .flat_map(|part| {
            fixture[part]["parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(k, v)| (k.clone(), v.clone()))
        })
        .collect();
    let (perception, rnnt) = dimensions();
    let mut model = PerceptionRnnt::new(perception, rnnt, context).unwrap();
    model.visit_parameters_mut(&mut Bind(parameters));
    model
}

fn limits(frame_capacity: u64, hypothesis_capacity: usize) -> PerceptionRnntLimits {
    PerceptionRnntLimits {
        frame_capacity,
        rnnt: RnntLimits {
            max_symbols_per_frame: 2,
            hypothesis_capacity,
        },
    }
}

fn assert_tensor(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let abs = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let rel = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (actual, expected) in actual.data.iter().zip(&expected.data) {
        assert!(
            (actual - expected).abs() <= abs + rel * expected.abs(),
            "{actual} != NVIDIA {expected}"
        );
    }
}

#[test]
fn voicechat_perception_rnnt_shares_raw_features_and_preserves_candidate_caches() {
    let fixture = fixture();
    let context = NumericContext::default();
    let mut model = model(&fixture, &context);
    let windows = fixture["subsampling"]["steps"].as_array().unwrap();
    let expected = fixture["perception"]["steps"].as_array().unwrap();
    let initial = PerceptionRnntState::default();
    // Capacity zero fails inside RNNT only after nonzero perception succeeds.
    // Neither that candidate nor partial predictor initialization is published.
    assert!(matches!(
        model.advance(
            &tensor(&windows[0]["input"]),
            0,
            &initial,
            limits(6, 0),
            &context
        ),
        Err(PerceptionRnntError::Rnnt(RnntError::Capacity {
            capacity: 0
        }))
    ));
    assert_eq!(initial.frames(), 0);
    assert!(initial.perception().blocks().is_empty());
    assert!(initial.rnnt().predictor().is_none());

    let first = model
        .advance(
            &tensor(&windows[0]["input"]),
            0,
            &initial,
            limits(6, 2),
            &context,
        )
        .unwrap();
    assert_eq!(first.emitted, [0, 0]);
    assert!(!first.first_blank && !first.ended_blank && first.hit_symbol_limit);
    assert_tensor(&first.encoder, &expected[0]["encoder_output"], &fixture);
    assert_tensor(&first.projected, &expected[0]["projected_output"], &fixture);
    assert!(
        first.state.rnnt().predictor().unwrap().local_state().layers[0]
            .hidden
            .data[0]
            > 0.0
    );
    let snapshot = first.state;
    for _ in 0..2 {
        let mut state = snapshot.clone();
        for (index, window) in windows.iter().enumerate().skip(1) {
            let result = model
                .advance(
                    &tensor(&window["input"]),
                    window["drop"].as_u64().unwrap() as usize,
                    &state,
                    limits(6, 2),
                    &context,
                )
                .unwrap();
            assert!(result.emitted.is_empty());
            assert!(result.first_blank && result.ended_blank && !result.hit_symbol_limit);
            assert_eq!(result.state.rnnt().hypothesis(), [0, 0]);
            assert_eq!(result.state.frames(), index as u64 + 1);
            assert_eq!(result.state.rnnt().frames(), index as u64 + 1);
            assert_tensor(
                &result.encoder,
                &expected[index]["encoder_output"],
                &fixture,
            );
            assert_tensor(
                &result.projected,
                &expected[index]["projected_output"],
                &fixture,
            );
            assert_tensor(
                result.state.perception().blocks()[0].channel().unwrap(),
                &expected[index]["channel_cache"],
                &fixture,
            );
            assert_eq!(state.frames(), index as u64);
            state = result.state;
        }
    }
    assert_eq!(snapshot.frames(), 1);
    assert_eq!(snapshot.rnnt().frames(), 1);
    assert_tensor(
        snapshot.perception().blocks()[0].channel().unwrap(),
        &expected[0]["channel_cache"],
        &fixture,
    );
}

#[test]
fn voicechat_perception_rnnt_rejects_geometry_and_bounds_before_neural_execution() {
    let context = NumericContext::default();
    let (perception, rnnt) = dimensions();
    assert!(matches!(
        PerceptionRnnt::<NumericBackend>::new(
            perception,
            RnntDimensions {
                encoder_width: 6,
                ..rnnt
            },
            &context
        ),
        Err(PerceptionRnntError::Geometry(_))
    ));
    // Leave all parameter slots unloaded: preflight must not inspect or execute them.
    let mut model = PerceptionRnnt::<NumericBackend>::new(perception, rnnt, &context).unwrap();
    let input = NumericTensor::new(vec![1, 1, 8], vec![0.2; 8]);
    let initial = PerceptionRnntState::default();
    assert!(matches!(
        model.advance(&input, 0, &initial, limits(0, 8), &context),
        Err(PerceptionRnntError::Capacity {
            required: 1,
            capacity: 0
        })
    ));
    assert!(matches!(
        model.advance(&input, 1, &initial, limits(6, 8), &context),
        Err(PerceptionRnntError::Geometry(_))
    ));
    let many = NumericTensor::new(vec![1, 17, 8], vec![0.2; 17 * 8]);
    assert!(matches!(
        model.advance(&many, 0, &initial, limits(6, 8), &context),
        Err(PerceptionRnntError::Geometry(_))
    ));
    let invalid = PerceptionRnntLimits {
        rnnt: RnntLimits {
            max_symbols_per_frame: 0,
            hypothesis_capacity: 8,
        },
        ..limits(6, 8)
    };
    assert!(matches!(
        model.advance(&input, 0, &initial, invalid, &context),
        Err(PerceptionRnntError::Rnnt(RnntError::InvalidConfiguration(
            _
        )))
    ));
    assert_eq!(context.embedding_forward_calls.get(), 0);
    assert_eq!(initial.frames(), 0);
}
