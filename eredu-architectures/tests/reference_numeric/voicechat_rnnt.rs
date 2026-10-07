use super::*;
use eredu_architectures::voicechat::rnnt::{
    Rnnt, RnntDimensions, RnntError, RnntExecution, RnntLimits, RnntPredictionState, RnntState,
};
use serde_json::Value;

fn values(value: &Value) -> Vec<f32> {
    value
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap() as f32)
        .collect()
}
pub(super) fn fixture() -> Value {
    serde_json::from_str(eredu_evaluation::voicechat::RNNT_REFERENCE_JSON).unwrap()
}
pub(super) fn dimensions(fixture: &Value) -> RnntDimensions {
    let d = &fixture["geometry"];
    RnntDimensions {
        vocab_size: d["vocab_size"].as_i64().unwrap() as i32,
        predictor_width: d["prediction_hidden"].as_i64().unwrap() as i32,
        predictor_layers: d["prediction_layers"].as_u64().unwrap() as usize,
        encoder_width: d["encoder_hidden"].as_i64().unwrap() as i32,
        joint_width: d["joint_hidden"].as_i64().unwrap() as i32,
    }
}
fn model(fixture: &Value, context: &NumericContext) -> Rnnt<NumericBackend> {
    struct Bind<'a>(&'a Value);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind<'_> {
        fn visit_mut(&mut self, metadata: ParameterMetadata, tensor: &'a mut NumericTensor) {
            let parameter = &self.0[metadata.id.as_str()];
            let shape: Vec<i32> = parameter["shape"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_i64().unwrap() as i32)
                .collect();
            assert_eq!(tensor.shape(), shape);
            *tensor = NumericTensor::new(shape, values(&parameter["values"]));
        }
    }
    let mut model = Rnnt::new(dimensions(fixture), context).unwrap();
    model.visit_parameters_mut(&mut Bind(&fixture["parameters"]));
    model
}
pub(super) fn assert_values(actual: &[f32], expected: &Value, fixture: &Value) {
    let expected = values(expected);
    assert_eq!(actual.len(), expected.len());
    let abs = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let rel = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (a, e) in actual.iter().zip(expected) {
        assert!((a - e).abs() <= abs + rel * e.abs(), "{a} != NVIDIA {e}");
    }
}
pub(super) fn assert_state(
    state: &RnntPredictionState<NumericTensor>,
    expected: &Value,
    fixture: &Value,
) {
    let hidden = state
        .local_state()
        .layers
        .iter()
        .flat_map(|l| l.hidden.data.iter().copied())
        .collect::<Vec<_>>();
    let cell = state
        .local_state()
        .layers
        .iter()
        .flat_map(|l| l.cell.data.iter().copied())
        .collect::<Vec<_>>();
    assert_values(&hidden, &expected["h"], fixture);
    assert_values(&cell, &expected["c"], fixture);
}
pub(super) fn frame(value: &Value) -> NumericTensor {
    NumericTensor::new(vec![1, 3], values(&value["encoder"]))
}
pub(super) fn limits(capacity: usize) -> RnntLimits {
    RnntLimits {
        max_symbols_per_frame: 3,
        hypothesis_capacity: capacity,
    }
}

#[test]
fn rnnt_prediction_joint_and_greedy_transitions_match_pinned_nvidia() {
    let fixture = fixture();
    let context = NumericContext::default();
    let mut model = model(&fixture, &context);
    let initial = RnntState::default();
    let mut prediction = model.prime(&initial, &context).unwrap();
    assert!(initial.predictor().is_none());
    assert!(initial.hypothesis().is_empty());
    assert_state(
        prediction.predictor().unwrap(),
        &fixture["predictor_steps"][0],
        &fixture,
    );
    for step in fixture["predictor_steps"]
        .as_array()
        .unwrap()
        .iter()
        .skip(1)
    {
        prediction = model
            .predict_token(
                step["token"].as_u64().unwrap() as usize,
                &prediction,
                32,
                &context,
            )
            .unwrap();
        assert_eq!(prediction.frames(), 0);
        assert_state(prediction.predictor().unwrap(), step, &fixture);
    }
    for case in ["streaming_frames", "feedback_terminates_frame"] {
        let mut state = model.prime(&initial, &context).unwrap();
        let mut all_tokens = Vec::new();
        for (index, expected) in fixture[case].as_array().unwrap().iter().enumerate() {
            let encoder = frame(expected);
            let logits = model.joint_logits(&encoder, &state, &context).unwrap();
            assert_values(
                &logits.data,
                &expected["joint_calls"][0]["logits"],
                &fixture,
            );
            let result = model
                .decode_frame(&encoder, &state, limits(32), &context)
                .unwrap();
            let tokens = expected["emitted"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<_>>();
            assert_eq!(result.emitted, tokens);
            assert_eq!(
                result.first_blank,
                expected["first_blank"].as_bool().unwrap()
            );
            assert_eq!(
                result.ended_blank,
                expected["ended_blank"].as_bool().unwrap()
            );
            assert_eq!(
                result.hit_symbol_limit,
                expected["hit_symbol_limit"].as_bool().unwrap()
            );
            all_tokens.extend(tokens);
            assert_eq!(result.state.hypothesis(), all_tokens);
            assert_eq!(result.state.frames(), index as u64 + 1);
            assert_state(result.state.predictor().unwrap(), expected, &fixture);
            state = result.state;
        }
    }
}

#[test]
fn rnnt_controlled_replay_fork_and_capacity_failure_preserve_parent_state() {
    let fixture = fixture();
    let context = NumericContext::default();
    let mut model = model(&fixture, &context);
    let frames = fixture["streaming_frames"].as_array().unwrap();
    let mut ordinary = RnntState::default();
    for input in frames {
        ordinary = model
            .decode_frame(&frame(input), &ordinary, limits(32), &context)
            .unwrap()
            .state;
    }
    let first = model
        .decode_frame(
            &frame(&frames[0]),
            &RnntState::default(),
            limits(32),
            &context,
        )
        .unwrap()
        .state;
    // The limit fails after two tentative recurrent updates; no parent state or
    // hypothesis is consumed, and the same frame can then be admitted normally.
    assert!(matches!(
        model.decode_frame(&frame(&frames[1]), &first, limits(2), &context),
        Err(RnntError::Capacity { capacity: 2 })
    ));
    assert_eq!(first.frames(), 1);
    assert!(first.hypothesis().is_empty());
    assert_state(first.predictor().unwrap(), &frames[0], &fixture);
    let snapshot = model
        .decode_frame(&frame(&frames[1]), &first, limits(32), &context)
        .unwrap()
        .state;
    for _ in 0..2 {
        let mut restored = snapshot.clone();
        for input in &frames[2..] {
            restored = model
                .decode_frame(&frame(input), &restored, limits(32), &context)
                .unwrap()
                .state;
        }
        assert_eq!(restored.hypothesis(), ordinary.hypothesis());
        assert_eq!(restored.frames(), ordinary.frames());
        assert_state(
            restored.predictor().unwrap(),
            frames.last().unwrap(),
            &fixture,
        );
    }
    let fork = model
        .decode_frame(&frame(&frames[3]), &snapshot, limits(32), &context)
        .unwrap();
    assert!(fork.emitted.is_empty());
    assert_eq!(fork.state.hypothesis(), snapshot.hypothesis());
    assert_state(snapshot.predictor().unwrap(), &frames[1], &fixture);
    assert!(matches!(
        model.predict_token(3, &snapshot, 32, &context),
        Err(RnntError::InvalidState(_))
    ));
}

#[test]
fn admitted_release_config_constructs_exact_rnnt_parameter_geometry() {
    let config = eredu_architectures::voicechat::config::VoiceChatConfig::from_json(include_str!(
        "../../src/voicechat/fixtures/official-inference-config.json"
    ))
    .unwrap();
    let dimensions = RnntDimensions::from_config(&config.rnnt).unwrap();
    assert_eq!(
        dimensions,
        RnntDimensions {
            vocab_size: 1024,
            predictor_width: 640,
            predictor_layers: 2,
            encoder_width: 1024,
            joint_width: 640
        }
    );
}
