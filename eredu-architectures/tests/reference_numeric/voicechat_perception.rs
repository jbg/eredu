use super::*;
use eredu_architectures::voicechat::{
    conformer::ConformerDimensions,
    perception::{Perception, PerceptionDimensions, PerceptionError, PerceptionState},
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

fn model(fixture: &Value, context: &NumericContext) -> Perception<NumericBackend> {
    struct Bind(BTreeMap<String, Value>);
    impl<'a> ParameterVisitorMut<'a, NumericTensor> for Bind {
        fn visit_mut(&mut self, metadata: ParameterMetadata, slot: &'a mut NumericTensor) {
            let expected = self.0.get(metadata.id.as_str()).unwrap_or_else(|| {
                panic!("independent fixture is missing {}", metadata.id.as_str())
            });
            let value = tensor(expected);
            assert_eq!(slot.shape(), value.shape(), "{}", metadata.id.as_str());
            *slot = value;
        }
    }
    let mut parameters = BTreeMap::new();
    for component in ["subsampling", "conformer", "perception"] {
        parameters.extend(
            fixture[component]["parameters"]
                .as_object()
                .unwrap()
                .iter()
                .map(|(name, value)| (name.clone(), value.clone())),
        );
    }
    let integer = |component: &str, field: &str| {
        fixture[component]["geometry"][field].as_i64().unwrap() as i32
    };
    let dimensions = PerceptionDimensions {
        subsampling: SubsamplingDimensions {
            features: integer("subsampling", "features"),
            channels: integer("subsampling", "channels"),
            width: integer("subsampling", "width"),
        },
        block: ConformerDimensions {
            width: integer("conformer", "width"),
            heads: integer("conformer", "heads"),
            ff_hidden: integer("conformer", "feed_forward"),
            conv_kernel: integer("conformer", "kernel"),
            left_context: integer("conformer", "left_context"),
        },
        layers: 1,
        output_width: fixture["perception"]["output_width"].as_i64().unwrap() as i32,
    };
    let mut model = Perception::new(dimensions, context).unwrap();
    model.visit_parameters_mut(&mut Bind(parameters));
    model
}

fn assert_tensor(actual: &NumericTensor, expected: &Value, fixture: &Value) {
    let expected = tensor(expected);
    assert_eq!(actual.shape(), expected.shape());
    let absolute = fixture["tolerance"]["absolute"].as_f64().unwrap() as f32;
    let relative = fixture["tolerance"]["relative"].as_f64().unwrap() as f32;
    for (index, (&actual, &expected)) in actual.data.iter().zip(&expected.data).enumerate() {
        assert!(
            (actual - expected).abs() <= absolute + relative * expected.abs(),
            "element {index}: {actual} != pinned NVIDIA {expected}"
        );
    }
}

fn assert_cache(
    state: &PerceptionState<NumericTensor>,
    expected: &Value,
    frames: u64,
    fixture: &Value,
    context: &NumericContext,
) {
    assert_eq!(state.frames(), frames);
    assert_eq!(state.blocks().len(), 1);
    let cache = &state.blocks()[0];
    assert_eq!(
        cache.valid_channel(),
        expected["valid_cache"].as_u64().unwrap() as usize
    );
    assert_tensor(
        cache.channel().unwrap(),
        &expected["channel_cache"],
        fixture,
    );
    // Portable convolution storage is time-major; upstream stores channel-major.
    let temporal = cache
        .convolution()
        .unwrap()
        .transpose_axes(&[0, 2, 1], context)
        .unwrap();
    assert_tensor(&temporal, &expected["time_cache"], fixture);
}

#[test]
fn voicechat_perception_full_and_cached_outputs_match_pinned_nvidia() {
    let fixture = fixture();
    let context = NumericContext::default();
    let mut model = model(&fixture, &context);
    let initial = PerceptionState::default();
    let full = model
        .forward(
            &tensor(&fixture["subsampling"]["input"]),
            0,
            &initial,
            &context,
        )
        .unwrap();
    assert_tensor(
        &full.encoder,
        &fixture["perception"]["encoder_output"],
        &fixture,
    );
    assert_tensor(
        &full.projected,
        &fixture["perception"]["projected_output"],
        &fixture,
    );
    let windows = fixture["subsampling"]["steps"].as_array().unwrap();
    let expected = fixture["perception"]["steps"].as_array().unwrap();
    assert_cache(&full.state, expected.last().unwrap(), 6, &fixture, &context);
    let mut state = PerceptionState::default();
    for (index, (window, expected)) in windows.iter().zip(expected).enumerate() {
        let next = model
            .forward(
                &tensor(&window["input"]),
                window["drop"].as_u64().unwrap() as usize,
                &state,
                &context,
            )
            .unwrap();
        assert_tensor(&next.encoder, &expected["encoder_output"], &fixture);
        assert_tensor(&next.projected, &expected["projected_output"], &fixture);
        assert_cache(&next.state, expected, index as u64 + 1, &fixture, &context);
        assert_eq!(state.frames(), index as u64);
        state = next.state;
    }
    assert!(initial.blocks().is_empty());
    assert_eq!(initial.frames(), 0);
}

#[test]
fn voicechat_perception_invalid_overlap_retry_snapshot_and_fork_preserve_caches() {
    let fixture = fixture();
    let context = NumericContext::default();
    let mut model = model(&fixture, &context);
    let windows = fixture["subsampling"]["steps"].as_array().unwrap();
    let expected = fixture["perception"]["steps"].as_array().unwrap();
    let mut prefix = PerceptionState::default();
    for window in &windows[..2] {
        prefix = model
            .forward(
                &tensor(&window["input"]),
                window["drop"].as_u64().unwrap() as usize,
                &prefix,
                &context,
            )
            .unwrap()
            .state;
    }
    let snapshot = prefix.clone();
    assert!(matches!(
        model.forward(&tensor(&windows[2]["input"]), 3, &snapshot, &context),
        Err(PerceptionError::State(_))
    ));
    assert_cache(&snapshot, &expected[1], 2, &fixture, &context);
    // Independently restored branches both continue against the external
    // oracle. Neither the rejected call nor a branch advances the snapshot.
    for _ in 0..2 {
        let mut branch = snapshot.clone();
        for (index, window) in windows.iter().enumerate().skip(2) {
            let next = model
                .forward(
                    &tensor(&window["input"]),
                    window["drop"].as_u64().unwrap() as usize,
                    &branch,
                    &context,
                )
                .unwrap();
            assert_tensor(&next.encoder, &expected[index]["encoder_output"], &fixture);
            assert_tensor(
                &next.projected,
                &expected[index]["projected_output"],
                &fixture,
            );
            assert_cache(
                &next.state,
                &expected[index],
                index as u64 + 1,
                &fixture,
                &context,
            );
            branch = next.state;
        }
        assert_cache(&snapshot, &expected[1], 2, &fixture, &context);
    }
}
